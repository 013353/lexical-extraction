PERIOD_LENGTH = 10

def head_file(filepath, header):
    """
    Clears the given file and adds a header to it, used for CSV file reset
    
    Parameters:
    -----------
    `filepath`: The file to head
    `header`: The header to add to the file
    """
    
    with open(filepath, "w") as file:
        file.write(header)
        
def clear_dir(directory):
    """
    Removes all files from the given directory
    
    Parameters:
    -----------
    `directory`: The directory to clear
    """
    
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def generate_profile(corpus, vectorizer, num):
    """
    Generates a profile of the given `corpus` with the middle 75% of ids
    
    Parameters:
    -----------
    `corpus`: The corpus from which to generate a profile
    `vectorizer`: The vectorizer to weight ids with, this program uses `TfidfVectorizer` and `CountVectorizer` from `sklearn.feature_extraction.text`
    `num`: The process number, used with multiprocessing to keep track of individual processes
    
    Returns:
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
    for index in token_df.index.to_list():
        completed = False
        while not completed:
            try:
                weights[index] = np.mean(token_df.loc[index])
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


def add_inputs_to_file(period, docs, filepath, tokenizer, tokenizer_params, vectorizer, num):
    """
    Adds `docs` to `file` with masks and `period` data
    
    Parameters:
    -----------
    `period`: The time period of `docs`
    `docs`: `pandas.Series` of the documents in the `period`
    `filepath`: The location of the file to write to
    `tokenizer`: The algorithm to use to tokenize `docs`
    `chunker_params` (`mode`, `size`): The set of parameters to pass to the chunker before tokenizing
        (`str`, `int`)
    `vectorizer`: The vectorizer to weight ids with, this program uses `TfidfVectorizer` and `CountVectorizer` from `sklearn.feature_extraction.text`
    `num`: The process number, used with multiprocessing to keep track of individual processes
    
    """
    
    # tokenize each document in the given period
    # returns a list of each document as a list
    tokenized_period = tokenize(docs, "sentence", 1, tokenizer)

    # clear file
    head_file(filepath, "doc;mask;period")

    # generate profile and append it to file
    profile = generate_profile(tokenized_period, vectorizer, num)
    with open(filepath, "a") as file:
        for doc in tokenized_period:
            mask = []
            for id in doc:
                mask.append(1 if id in profile else 0)
            file.write(f"\n{doc};{mask};{period}")

    
def tokenize(dataset, mode, size, model):
    """
    Tokenizes documents from `dataset`
    
    Parameters:
    -----------
    `dataset`: `pandas.Series` of the documents to tokenize
    `mode` (`"sentence"`, `"paragraph"`, `"word"`): The mode to chunk each document with
    `size`: The size of each chunk
    `model`: The algorithm to use to tokenize each document
    
    Returns:
    --------
    `list`: The chunked documents as a 2D list
    
    """
    
    # create a list of docs separated into chunks
    chunked_docs = []
    for i, row in dataset.iterrows():
        chunked_docs.append(chunk_file(row["filepath"], mode, size))
    
    # create a list of each chunk, tokenized
    tokenized_chunks = []
    for doc in chunked_docs:
        for chunk in doc:
            
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


def separate_periods(df):
    """
    Separates docs in `df` by period
    
    Parameters:
    -----------
    `df`: The dataframe to separate
        `columns` = [`"filepath"`, `"title"`, `"year"`]
    
    Returns:
    --------
    `dict`: The documents separated by period
        `key`: Period
        `value`: `pandas.DataFrame` of docs in the period
            `columns` = [`"filepath"`, `"title"`, `"year"`]

    """
    
    periods = {}
    
    # sort docs by year, print result
    docs_df = df.sort_values(by="year")
    print(docs_df)
    
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

def transform_test(tokenized_docs, model, output_file):
    # torch.no_grad() disables gradient calculation to prevent OOM error
    with torch.no_grad():
        # find first and last indices of batch in test data
        first = np.floor(BATCH_SIZE * i)
        last = np.floor(BATCH_SIZE * (i+1))
        
        # convert test docs to GPU tensor
        docs = torch.tensor(tokenized_docs[first:last], device=dev)
        
        # pass test docs through model, get pooler_output
        output = model.forward(input_ids=docs).pooler_output.tolist()
        
        # add outputs to file
        for i in range(BATCH_SIZE):
            output_file.write(f"\n{output[i]};{test_years[first+i]}")


def get_accuracy(estimate, expected):
    """
    Returns a tuple of bools of if the `estimate` is within each measure of `expected`
    
    `Acc@K` is a measure of accuracy that considers all documents within K/2 periods of expected to be accurate (Ren et al., 2023)
    
    Parameters:
    -----------
    `estimate`: The estimated period
    `expected`: The actual period
    
    Returns:
    --------
    `tuple`: (`acc`, `acc@3`, `acc@5`)
        (`bool`, `bool`, `bool`)
    
    """
    
    acc, acc_3, acc_5 = False
    if estimate == expected: acc = True
    if expected-PERIOD_LENGTH <= estimate <= expected+PERIOD_LENGTH: acc_3 = True
    if expected-2*PERIOD_LENGTH <= estimate <= expected+2*PERIOD_LENGTH: acc_5 = True
    
    return acc, acc_3, acc_5


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

if __name__ == "__main__":
    print("\rDONE!")

    # retrieve doc data from file
    data_df = pd.read_csv("Documents/_doc_data.csv")

    # activate PyTorch cuda support if available
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = "cpu"
    print("Device:", dev)

    # split docs into train and test data
    train, test = train_test_split(data_df, train_size=0.5)

    # initialize tf-idf and standard vectorizers
    vectorizers = [CountVectorizer()]

    # list chunker parameter combinations
    chunker_params = [("paragraph", 1), ("sentence", 5), ("word", 100), ("word", 200), ("sentence", 10), ("paragraph", 2), ("word", 300)]
    # chunker_params = [("sentence", 5), ("sentence", 10), ("paragraph", 1), ("paragraph", 2), ("word", 100), ("word", 200), ("word", 300)]

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
            
            for chunker_params in chunker_params:
            
                for vectorizer in vectorizers:

                    # delete all train_docs and profile files
                    clear_dir("FR_Test/train_docs")
                    clear_dir("FR_Test/profiles")
                    
                    # start threads to generate each profile, store threads in processes
                    processes = []
                    i=0
                    for key, value in tqdm(separate_periods(train).items(), desc="Starting Profile Generators"):
                        process = mp.Process(target=add_inputs_to_file, args=(key, value, f"FR_Test/train_docs/train_docs_{i}.csv", tokenizer, chunker_params, vectorizer, i))
                        processes.append(process)
                        process.start()
                        i+=1
                    
                    # join profile generator threads and add docs to train_docs dataframe
                    train_docs = pd.DataFrame(columns=["doc", "mask", "period"])
                    for index in tqdm(range(len(processes)), desc="Waiting for Profile Generators"):
                        processes[index].join()
                        df = pd.read_csv(f"FR_Test/train_docs/train_docs_{index}.csv", sep=";")
                        train_docs = pd.concat([train_docs, df], ignore_index=True)
                    
                    for index in tqdm(range(23)):
                        df = pd.read_csv(f"FR_Test/train_docs/train_docs_{index}.csv", sep=";")
                        train_docs = pd.concat([train_docs, df], ignore_index=True)
                    
                    # del processes
                        
                    def match_lengths(col, length):
                        """
                        Sets the length of `col` of `train_docs` to a set length
                        
                        Parameters:
                        -----------
                        `col`: The column to edit
                        `length`: The length to adjust to
                        
                        """
                        
                        # get the specified column of train_docs as a series
                        series = train_docs.loc[:, col]
                        
                        # set the length of each list in the column to len, truncate or pad with 0s if necessary
                        for i in tqdm(range(len(series)), leave=False):
                            ls = eval(train_docs.at[i, col])
                            train_docs.at[i, col] = ls[:length] + [0]*(length-len(ls))

                    # set all docs and masks to the same length
                    match_lengths("doc", max_input_length)
                    match_lengths("mask", max_input_length)
                    
                    train_docs.to_csv("FR_Test/train_docs.csv", sep=";")
                    
                    train_docs = pd.read_csv("FR_Test/train_docs.csv")
                    print(train_docs)
                    train_docs["doc"] = train_docs["doc"].apply(lambda x: eval(x))
                    train_docs["mask"] = train_docs["mask"].apply(lambda x: eval(x))
                    print("DONE")
                    train_docs = pd.read_csv("FR_Test/train_docs.csv")
                    
                    # shuffle train_docs
                    train_docs = train_docs.sample(frac=1).reset_index(drop=True)
                    
                    # initialize SVM
                    svm = LinearSVC()

                    BATCH_SIZE = 128
                    
                    # clear train_outputs.csv
                    head_file("FR_Test/train_outputs.csv", "output;period")
                    
                    with open("FR_Test/train_outputs.csv", "a") as train_outputs:

                        # pass all docs through the model, batch size specified above
                        NUM_BATCHES_TRAIN  = int(np.ceil(len(train_docs.index)/BATCH_SIZE))
                        for batch in tqdm(range(NUM_BATCHES_TRAIN), desc=transformer):
                            
                            # torch.no_grad() disables gradient calculation to prevent OOM error
                            with torch.no_grad():
                                
                                # find first and last indices of batch in train_docs
                                first = np.floor(BATCH_SIZE * batch)
                                last = np.floor(BATCH_SIZE * (batch+1)) - 1
                                if last > len(train_docs.index):
                                    last = len(train_docs.index)
                                
                                # convert train docs and masks to GPU tensors
                                docs = torch.tensor(train_docs.loc[first:last, "doc"].tolist(), device=dev)
                                masks = torch.tensor(train_docs.loc[first:last, "mask"].tolist(), device=dev)
                                
                                # pass tensors into model, get pooler_output
                                output = model.forward(input_ids=docs, attention_mask=masks).pooler_output.tolist()
                                
                                # print(len(output) == BATCH_SIZE)
                                
                                # add outputs to file
                                for i in range(BATCH_SIZE):
                                    train_outputs.write(f"\n{output[i]};{train_docs.loc[first+i, "period"]}")
                    
                    # create a dataframe from the outputs of the model
                    print("Reading Train Outputs", end="")
                    train_outputs_df = pd.read_csv("FR_Test/train_outputs.csv", sep=";")
                    print(".", end="")
                    train_outputs_df = train_outputs_df.sample(frac=0.25).reset_index(drop=True)
                    print(".", end="")
                    train_outputs_df["output"] = train_outputs_df["output"].apply(lambda x: eval(x))
                    print(". DONE!")
                    
                    # train the SVM om the outputs
                    print("Training SVM...", end="")
                    svm.fit(np.array(train_outputs_df["output"].values.tolist()), np.array(train_outputs_df["period"].values.tolist()))
                    del train_outputs_df
                    print("DONE!")
                    
                    # tokenize test data and store years in test_years
                    tokenized_test = tokenize(test, "sentence", 1, tokenizer)
                    test_years = test["year"].values.tolist()
                    print(tokenized_test)
                    print(test_years)
                    
                    test_outputs = []
                    
                    # clear test_outputs.csv
                    head_file("FR_Test/test_outputs.csv", "output;period")
                    
                    processes = []
                    
                    with open("FR_Test/test_outputs.csv", "a") as test_outputs:
                        
                        # pass all test documents through model
                        NUM_BATCHES_TEST  = int(np.ceil(len(test.index)/BATCH_SIZE))
                        for i in tqdm(range(NUM_BATCHES_TEST), desc="Starting BERT Test"):

                            process = mp.Process(target=transform_test, args=(tokenized_test, model, test_outputs))
                            processes.append(process)
                            process.start()
                        
                    for process in tqdm(processes, desc="Waiting for BERT"):
                        process.join()


                    # create a dataframe from the outputs of the model
                    test_outputs_df = pd.read_csv("FR_Test/test_outputs.csv", sep=";")
                    test_outputs_df = test_outputs_df.sample(frac=0.25).reset_index(drop=True)
                    test_outputs_df["output"] = test_outputs_df["output"].apply(lambda x: eval(x))
                        
                    
                    # get estimates of the year of each test document from the SVM
                    estimates = svm.predict(test_outputs_df.loc[:, "output"])
                    
                    # assess the accuracy of the model using Acc, Acc@3, and Acc@5
                    acc_data, acc_3_data, acc_5_data = []
                    for i in tqdm(range(len(estimates))):
                        estimate = estimates.item(i)
                        expected = test_years[i]
                        accs = get_accuracy(estimate, expected)
                        acc_data.append(accs[0])
                        acc_3_data.append(accs[1])
                        acc_5_data.append(accs[2])
                    
                    # identify the mean accuracy of the model for each metric
                    acc = np.mean(acc)
                    acc_3 = np.mean(acc_3)
                    acc_5 = np.mean(acc_5)
                    
                    # print results
                    print("Acc:", acc)
                    print("Acc@3:", acc_3)
                    print("Acc@5:", acc_5)
                    
                    # add results to results.csv
                    head_file("FR_Test/results.csv", "transformer;chunker_params;vectorizer;acc;acc@3;acc@5")
                    result_line = transformer + ";" + str(chunker_params) + ";"
                    match vectorizer:
                        case TfidfVectorizer():
                            result_line += "tf-idf"
                        case CountVectorizer():
                            result_line += "count"
                    result_line += ";" + acc + ";" + acc_3 + ";" + acc_5
                    with open("FR_Test/results.csv", "a") as results_file:
                        results_file.write(result_line)

    # print overall test results
    print(pd.read_csv("FR_Test/results.csv"))