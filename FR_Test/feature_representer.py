if __name__ == "__main__":
    print("Importing packages... ")

completed = False
while not completed:
    try:
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.svm import LinearSVC
        from transformers import BertTokenizerFast, BertModel, RobertaTokenizerFast, RobertaModel, LongformerTokenizerFast, LongformerModel
        from transformers.utils import logging
        import torch
        import multiprocessing as mp
        import time
        import re
        import os
        from chunker import chunk_file
        completed = True
    except OSError as e:
        print(e)

# set transformers to only print errors to the console
logging.set_verbosity_error()

# make pandas use tqdm progress bars
tqdm.pandas()

PERIOD_LENGTH = 10

def head_file(filepath : str, 
              header: str
              ) -> None:
    """
    Clears the given file and adds a header to it, used for CSV file reset
    
    Parameters
    -----------
    `filepath`: The file to head\n
    `header`: The header to add to the file
    """
    
    with open(filepath, "w") as file:
        file.write(header)
        
def clear_dir(directory : str
              ) -> None:
    """
    Removes all files from the given directory
    
    Parameters
    -----------
    `directory`: The directory clear
    """
    
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def generate_profile(corpus : list,
                     vectorizer : TfidfVectorizer | CountVectorizer,
                     num : int
                     ) -> list:
    """
    Generates a profile of the given `corpus` with the middle 75% of ids
    
    Parameters
    -----------
    `corpus`: The corpus from which to generate a profile\n
    `vectorizer`: The vectorizer to weight ids with, this program uses `TfidfVectorizer` and `CountVectorizer` from `sklearn.feature_extraction.text`\n
    `num`: The process number, used with multiprocessing to keep track of individual processes
    
    Returns
    --------
    `list`: Middle 75% of ids, sorted from highest to lowest weight
    """
    
    # convert each doc from a list of ints to a list of strings
    string_corpus = []
    for doc in corpus:
        doc_string = ""
        for token in doc:
            doc_string += str(token) if len(doc_string) == 0 else " " + str(token)
        string_corpus.append(doc_string)
    
    # vectorize each doc when memory is available
    completed = False
    while not completed:
        try:
            arr = vectorizer.fit_transform(string_corpus)
            del string_corpus
            completed = True
        except MemoryError as e:
            print(num, e)
            time.sleep(10)

    # create a dataframe of the tf-idf score of each token in each document
    token_df = pd.DataFrame.sparse.from_spmatrix(arr, columns=vectorizer.get_feature_names_out())

    del arr
    
    # create a dict of the mean tf-idf score of each token
    weights = {}
    for col in token_df.columns.to_list():
        completed = False
        while not completed:
            try:
                weights[col] = np.mean(token_df.loc[:, col])
                completed = True
            except MemoryError:
                time.sleep(30)
                
    del token_df
    
    # convert mean weight dict to dataframe and sort by weight
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=["weight"])
    sorted_weights_df = weights_df.sort_values(by=weights_df.columns[0], axis=0, ascending=False)
    del weights_df
    
    # find the mean and standard deviation of the set of integers with the same length as sorted_weights_df
    #TODO make this more succinct
    weights_x = []
    for i, row in sorted_weights_df.iterrows():
        weights_x.append(row["weight"])
    weights_mean = np.mean(range(len(weights_x)))
    weights_std = np.std(range(len(weights_x)))
    
    # # maybe this is better?
    # weights_mean = np.mean(weights_x)
    # weights_std = np.std(weights_x)
    
    def bell_curve(x, std, mean, mult=1):
        return mult/(std * np.sqrt(2 * np.pi)) * np.e**( - (x - mean)**2 / (2 * std**2))
    
    # weight each weight according to a bell curve raised to the 4th power
    i = 0
    for index, row in sorted_weights_df.iterrows():
        i += 1
        sorted_weights_df.at[index, "weight"] = bell_curve(i, weights_std, weights_mean)**4 * row["weight"]
    
    # make the minimum weight 1
    mult = 1/min(list(sorted_weights_df["weight"]))
    for index, row in sorted_weights_df.iterrows():
        sorted_weights_df.at[index, "weight"] *= mult
    
    # re-sort the weights
    sorted_weights_df = sorted_weights_df.sort_values(by=sorted_weights_df.columns[0], axis=0, ascending=False)
    
    # identify the upper and lower quantile boundaries
    bottom = sorted_weights_df.quantile(0.125)["weight"]
    top = sorted_weights_df.quantile(0.875)["weight"]
    
    # remove weights outside the selected quantile range
    middle_weights = sorted_weights_df[sorted_weights_df["weight"] < top]
    middle_weights = middle_weights[middle_weights["weight"] > bottom]
    
    # identify the ids in the selected range, save to file, return
    middle_ids = list(middle_weights.index)
    sorted_weights_df.to_csv(f"FR_Test/profiles/{num}.csv")
    
    return middle_ids


def add_inputs_to_file(period : int,
                       docs : any,
                       filepath : str,
                       tokenizer : any,
                       tokenizer_params : tuple[str, int],
                       vectorizer : TfidfVectorizer | CountVectorizer,
                       num : int,
                       generate_profile_and_mask : bool 
                        ) -> None:
    """
    Adds `docs` to `file` with masks and `period` data
    
    Parameters
    -----------
    `period`: The time period of `docs`\n
    `docs`: `pandas.Series` of the documents in the `period`\n
    `filepath`: The location of the file to write to\n
    `tokenizer`: The algorithm to use to tokenize `docs`\n
    `tokenizer_params` (`mode`, `size`): The set of parameters to pass to the chunker before tokenizing\n
    `vectorizer`: The vectorizer to weight ids with, this program uses `TfidfVectorizer` and `CountVectorizer` from `sklearn.feature_extraction.text`\n
    `num`: The process number, used with multiprocessing to keep track of individual processes
    """
    
    # tokenize each document in the given period
    # returns a list of each document as a list
    tokenizer_mode = tokenizer_params[0]
    tokenizer_size = tokenizer_params[1]
    tokenized_period = tokenize(docs, tokenizer_mode, tokenizer_size, tokenizer)

    # clear file
    head_file(filepath, "doc;mask;period")

    if generate_profile_and_mask:
        # generate profile and append it to file
        profile = generate_profile(tokenized_period, vectorizer, num)
        with open(filepath, "a") as file:
            for doc in tokenized_period:
                mask = []
                for id in doc:
                    mask.append(1 if id in profile else 0)
                file.write(f"\n{doc};{mask};{period}")
    else:
        # add masks of 1s of same length as chunks
        with open(filepath, "a") as file:
            for doc in tokenized_period:
                mask = [1]*len(doc)
                file.write(f"\n{doc};{mask};{period}")

    
def tokenize(dataset : any, 
             mode : str, 
             size : int, 
             model: any
             ) -> list:
    """
    Tokenizes documents from `dataset`
    
    Parameters
    -----------
    `dataset`: `pandas.Series` of the documents to tokenize\n
    `mode` (`"sentence"`, `"paragraph"`, `"word"`): The mode to chunk each document with\n
    `size`: The size of each chunk\n
    `model`: The algorithm to use to tokenize each document
    
    Returns
    --------
    `list`: The chunked documents as a 2D list
    
    """
    
    # create a list of docs separated into chunks
    chunked_docs = []
    for i, row in dataset.iterrows():
        chunked_docs.extend(chunk_file(row["filepath"], mode, size))

    # take a random 25000 chunks from chunked_docs
    chunked_docs = sample_chunks(chunked_docs, 25000)
    
    # create a list of each chunk, tokenized
    tokenized_chunks = []
    for chunk in chunked_docs:
        # split each chunk into words
        words = chunk.split()
        
        # filter out inaccurately digitized words into a list
        masked_chunk = []
        for word in words:
            if re.match(r"\w+", word):
                masked_chunk.append(word)
            else:
                masked_chunk.append("[UNK]")
        
        # reform the list of words into a string
        chunk = " ".join(masked_chunk)
        
        # tokenize the chunk and append to tokenized_chunks
        tokenized_chunk = model.encode(chunk)
        tokenized_chunks.append(tokenized_chunk)

    return tokenized_chunks


def separate_periods(df : pd.DataFrame
                     ) -> dict[int, pd.DataFrame]:
    """
    Separates docs in `df` by period
    
    Parameters
    -----------
    `df`: The dataframe to separate\n
        `columns` = [`"filepath"`, `"title"`, `"year"`]
    
    Returns
    --------
    `dict`: The documents separated by period\n
        `key`: Period\n
        `value`: `pandas.DataFrame` of docs in the period\n
            `columns` = [`"filepath"`, `"title"`, `"year"`]

    """
    
    periods = {}
    
    # sort docs by year, print result
    docs_df = df.sort_values(by="year")
    # print(docs_df)
    
    # separate docs_df into periods and add docs as a dataframe to periods
    for index, row in tqdm(docs_df.iterrows(), total=len(docs_df.index), desc="Separating Documents"):
        
        # find the period of the doc rounded down
        period = int(PERIOD_LENGTH * np.floor(row["year"]/PERIOD_LENGTH))
        
        # create a new dataframe in periods if one doesn't already exist
        if not period in periods:
            periods[period] = pd.DataFrame(columns=["filepath", "title", "year"])
            
        # add the current doc data to the period dataframe in periods
        r = pd.DataFrame(row).transpose()
        periods[period].loc[len(periods[period])] = r.iloc[0]

    return periods

def sample_chunks(input : list,
                  num : int
                  ) -> list:
    """Samples `num` chunks from the `input` list of chunks

    Args:
        input (list): The list of chunks to sample from
        num (int): The number of chunks to sample from `input`

    Returns:
        list: The sampled chunks
    """

    return pd.Series(input).sample(frac=1, ignore_index=True).truncate(after=num-1).tolist()

def get_accuracy(estimate : int,
                 expected : int
                 ) -> tuple[bool, bool, bool]:
    """
    Returns a tuple of bools of if the `estimate` is within each measure of `expected`
    
    `Acc@K` is a measure of accuracy that considers all documents within K/2 periods of expected to be accurate (Ren et al., 2023)
    
    Parameters
    -----------
    `estimate`: The estimated period\n
    `expected`: The actual period
    
    Returns
    --------
    `tuple`: (`acc`, `acc@3`, `acc@5`)
        (`bool`, `bool`, `bool`)
    
    """
    
    acc = acc_3 = acc_5 = 0
    if estimate == expected: acc = 1
    if expected-PERIOD_LENGTH <= estimate <= expected+PERIOD_LENGTH: acc_3 = 1
    if expected-2*PERIOD_LENGTH <= estimate <= expected+2*PERIOD_LENGTH: acc_5 = 1
    
    with open("FR_Test/confusion.csv", "a") as file:
        file.write(f"\n{estimate};{expected}")

    return acc, acc_3, acc_5


def match_lengths(col : str, 
                  length : int,
                  df : pd.DataFrame
                  ) -> None:
    """
    Sets the length of `col` of `df` to a set length
    
    Parameters
    -----------
    `col`: The column to edit\n
    `length`: The length to adjust to\n
    `df`: The  `pandas.DataFrame` to adjust
    """
    
    # get the specified column of `df` as a series
    series = df.loc[:, col]
    
    # set the length of each list in the column to len, truncate or pad with 0s if necessary
    for i in tqdm(range(len(series)), leave=False):
        ls = eval(df.at[i, col])
        df.at[i, col] = ls[:length] + [0]*(length-len(ls))

def transformer_model(data : pd.DataFrame,
                      model : any,
                      output_filepath : str
                      ) -> pd.DataFrame:
    """Runs the given transformer model on the given data

    Args:
        data (pd.DataFrame): the data to run on
        transformer (any): the transformer model to use, such as `BERT` or `RoBERTa` or `Longformer`
        output_filepath (str): the file to print output data to

    Returns:
        pd.DataFrame: the output data of the transformer
    """

    with open(output_filepath, "a") as output_file:

        # pass all docs through the model, batch size specified above
        NUM_BATCHES  = int(np.ceil(len(data.index)/BATCH_SIZE))
        for batch in tqdm(range(NUM_BATCHES), desc=transformer):
            
            # torch.no_grad() disables gradient calculation to prevent OOM error
            with torch.no_grad():
                
                # find first and last indices of batch in data
                first = np.floor(BATCH_SIZE * batch)
                last = np.floor(BATCH_SIZE * (batch+1)) - 1
                if last > len(data.index):
                    last = len(data.index)
                
                # convert test docs and masks to GPU tensors
                docs = torch.tensor(data.loc[first:last, "doc"].tolist(), device=dev)
                masks = torch.tensor(data.loc[first:last, "mask"].tolist(), device=dev)
                
                # pass tensors into model, get pooler_output
                output = model.forward(input_ids=docs, attention_mask=masks).pooler_output.tolist()
                
                # print(len(output) == BATCH_SIZE)
                
                # add outputs to file
                for i in range(last-first):
                    output_file.write(f"\n{output[i]};{data.loc[first+i, "period"]}")

    # create a dataframe from the outputs of the model
    outputs_df = pd.read_csv("FR_Test/test_outputs.csv", sep=";")
    # outputs_df = outputs_df.sample(frac=0.25).reset_index(drop=True)
    outputs_df["output"] = outputs_df["output"].apply(lambda x: eval(x))

    return outputs_df


if __name__ == "__main__":
    print("\rDONE!")

    # retrieve doc data from file
    data_df = pd.read_csv("Documents/_doc_data.csv")

    # activate PyTorch cuda support if available
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", dev)

    # split docs into train and test data
    train, test = train_test_split(data_df, train_size=0.8)

    # # clear and head results file
    # head_file("FR_Test/results.csv", "transformer;chunker_params;model;acc;acc@3;acc@5")

    completed_tests = [("BERT", ("paragraph", 1), True)]

    # list chunker parameter combinations
    chunker_params_list = [("paragraph", 1), ("sentence", 3), ("word", 100), ("word", 200), ("sentence", 5), ("paragraph", 2)]

    transformers = ["BERT", "RoBERTa", "Longformer"]    
    for transformer in transformers:
        
        # disables gradient calculation, saving GPU memory
        with torch.no_grad():

            # initialize tokenizers and models
            match transformer:
                case "BERT":
                    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
                    model = BertModel.from_pretrained("google-bert/bert-base-uncased").to(dev)
                    max_input_length = 512
                case "RoBERTa":
                    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
                    model = RobertaModel.from_pretrained("FacebookAI/roberta-base").to(dev)
                    max_input_length = 512
                case "Longformer":
                    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
                    model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(dev)
                    max_input_length = 1024
            
            for chunker_params in chunker_params_list:
            
                for lexical_extraction in [True, False]:

                    name = transformer[0] + chunker_params[0][0].upper() + str(chunker_params[1]) + ("T" if lexical_extraction else "F")

                    if not (transformer, chunker_params, lexical_extraction) in completed_tests:

                        print("Name:", name)

                        # initialize vectorizer
                        vectorizer = TfidfVectorizer() if lexical_extraction else CountVectorizer()

                        # delete all train_data and profile files
                        clear_dir("FR_Test/train_data")
                        clear_dir("FR_Test/profiles")
                        
                        # start threads to generate each profile, store threads in processes
                        processes = []
                        i=0
                        for key, value in tqdm(separate_periods(train).items(), desc="Starting Profile Generators"):
                            process = mp.Process(target=add_inputs_to_file, args=(key, value, f"FR_Test/train_data/train_data_{i}.csv", tokenizer, chunker_params, vectorizer, i, lexical_extraction))
                            processes.append(process)
                            process.start()
                            i+=1
                        
                        # join profile generator threads and add docs to train_data dataframe
                        train_data = pd.DataFrame(columns=["doc", "mask", "period"])
                        for index in tqdm(range(len(processes)), desc="Waiting for Profile Generators"):
                            processes[index].join()
                            df = pd.read_csv(f"FR_Test/train_data/train_data_{index}.csv", sep=";")
                            train_data = pd.concat([train_data, df], ignore_index=True)
                        
                        del processes

                        # set all docs and masks to the same length
                        match_lengths("doc", max_input_length, train_data)
                        match_lengths("mask", max_input_length, train_data)
                        
                        # save train_data to a CSV file and serialize as a pickle
                        train_data.to_csv("FR_Test/train_data.csv", sep=";")
                        train_data.to_pickle("FR_Test/train_data.pickle")
                    
                    
                        # shuffle train_data
                        train_data = train_data.sample(frac=1).reset_index(drop=True)

                        BATCH_SIZE = 512
                        
                        # clear train_outputs.csv
                        head_file("FR_Test/train_outputs.csv", "output;period")
                        
                        # pass the train data through the transformer model and save to CSV
                        train_outputs_df = transformer_model(train_data, model, "FR_Test/train_outputs.csv")

                        # serialize transformer model output
                        train_outputs_df.to_pickle("FR_Test/train_outputs.pickle")

                        del train_outputs_df
                        
                        # start threads to generate each profile, store threads in processes
                        processes = []
                        i=0
                        for key, value in tqdm(separate_periods(test).items(), desc="Starting Profile Generators"):
                            process = mp.Process(target=add_inputs_to_file, args=(key, value, f"FR_Test/test_data/test_data_{i}.csv", tokenizer, chunker_params, vectorizer, i, False))
                            processes.append(process)
                            process.start()
                            i+=1
                        
                        # join profile generator threads and add docs to test_data dataframe
                        test_data = pd.DataFrame(columns=["doc", "mask", "period"])
                        for index in tqdm(range(len(processes)), desc="Waiting for Profile Generators"):
                            processes[index].join()
                            df = pd.read_csv(f"FR_Test/test_data/test_data_{index}.csv", sep=";")
                            test_data = pd.concat([test_data, df], ignore_index=True)
                        
                        del processes

                        # set all docs and masks to the same length
                        match_lengths("doc", max_input_length, test_data)
                        match_lengths("mask", max_input_length, test_data)
                        
                        # shuffle test_data
                        test_data = test_data.sample(frac=1).reset_index(drop=True)

                        # save test data to CSV and serialize
                        test_data.to_csv("FR_Test/test_data.csv", sep=";")
                        test_data.to_pickle("FR_Test/test_data.pickle")
                        
                        # clear test_outputs.csv
                        head_file("FR_Test/test_outputs.csv", "output;period")
                        
                        test_outputs_df = transformer_model(test_data, model, "FR_Test/test_outputs.csv")
                        test_outputs_df.to_pickle("FR_Test/test_outputs.pickle")
        
                        # read train_outputs from pickle
                        train_outputs_df = pd.read_pickle("FR_Test/train_outputs.pickle")

                        # initialize SVM
                        svm = LinearSVC(max_iter=10000000)
                        
                        # train the SVM om the training outputs
                        svm_start_time = time.time()
                        print("Training SVM... ", end="")
                        svm.fit(np.array(train_outputs_df["output"].values.tolist()), np.array(train_outputs_df["period"].values.tolist()))
                        print("DONE! (" + time.strftime("%H:%M:%S", time.gmtime(time.time()-svm_start_time)) + ")")    

                        del train_outputs_df

                        test_outputs_df = pd.read_pickle("FR_Test/test_outputs.pickle")
                        
                        # get estimates of the year of each test document from the SVM
                        svm_start_time = time.time()
                        print("Generating Estimates... ", end="")
                        estimates = svm.predict(test_outputs_df["output"].tolist())
                        print("DONE! (" + time.strftime("%H:%M:%S", time.gmtime(time.time()-svm_start_time)) + ")")    
                        
                        # assess the accuracy of the model using Acc, Acc@3, and Acc@5
                        expected_years = test_outputs_df.loc[:, "period"].values.tolist()

                        del test_outputs_df
                        
                        head_file("FR_Test/confusion.csv", "estimate;expected")
                        acc_data, acc_3_data, acc_5_data = ([],[],[])
                        for i in tqdm(range(len(estimates)), desc="Evaluating Accuracy"):
                            estimate = estimates.item(i)
                            expected = expected_years[i]
                            accs = get_accuracy(estimate, expected)
                            acc_data.append(accs[0])
                            acc_3_data.append(accs[1])
                            acc_5_data.append(accs[2])
                        
                        # identify the mean accuracy of the model for each metric
                        acc = np.mean(acc_data)
                        acc_3 = np.mean(acc_3_data)
                        acc_5 = np.mean(acc_5_data)
                        
                        # print results
                        print("Acc:", acc)
                        print("Acc@3:", acc_3)
                        print("Acc@5:", acc_5)
                        
                        # add results to results.csv
                        result_line = transformer + ";" + str(chunker_params) + ";"
                        result_line += "LE" if lexical_extraction else "BASE"
                        match lexical_extraction:
                            case TfidfVectorizer():
                                result_line += "tf-idf"
                            case CountVectorizer():
                                result_line += "count"
                        result_line += ";" + str(acc) + ";" + str(acc_3) + ";" + str(acc_5)
                        with open("FR_Test/results.csv", "a") as results_file:
                            results_file.write("\n" + result_line)

                        # print overall test results
                        print("============================================================================")
                        print(pd.read_csv("FR_Test/results.csv", sep=";"))
                        print("============================================================================")