PERIOD_LENGTH = 10

def generate_profile(corpus, vectorizer, num):
    string_corpus = []
    for doc in corpus:
        doc_string = ""
        for token in doc:
            doc_string += str(token) if len(doc_string) == 0 else " " + str(token)
        string_corpus.append(doc_string)
        
    completed = False
    while not completed:
        try:
            arr = vectorizer.fit_transform(string_corpus)
            del string_corpus
            completed = True
        except MemoryError as e:
            print(num, e)
            time.sleep(10)

    token_df = pd.DataFrame.sparse.from_spmatrix(arr, columns=vectorizer.get_feature_names_out())

    del arr
    
    weights = {}
    for index in token_df.index.to_list():
        weights[index] = token_df.loc[index].mean()
    del token_df
    
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=["weight"])
    
    sorted_weights_df = weights_df.sort_values(by=weights_df.columns[0], axis=0, ascending=False)
    del weights_df
    
    weights_x = []
    
    for i, row in sorted_weights_df.iterrows():
        weights_x.append(row["weight"])
    
    weights_mean = np.mean(range(len(weights_x)))
    weights_std = np.std(range(len(weights_x)))
    
    def bell_curve(x, std, mean, mult=1):
        return mult/(std * np.sqrt(2 * np.pi)) * np.e**( - (x - mean)**2 / (2 * std**2))
    
    i = 0
    for index, row in sorted_weights_df.iterrows():
        i += 1
        sorted_weights_df.at[index, "weight"] = bell_curve(i, weights_std, weights_mean)**4 * row["weight"]
        
    mult = 1/min(list(sorted_weights_df["weight"]))
    
    for index, row in sorted_weights_df.iterrows():
        sorted_weights_df.at[index, "weight"] *= mult
    
    sorted_weights_df = sorted_weights_df.sort_values(by=sorted_weights_df.columns[0], axis=0, ascending=False)
        
    bottom = sorted_weights_df.quantile(0.125)["weight"]
    top = sorted_weights_df.quantile(0.875)["weight"]
    
    middle_weights = sorted_weights_df[sorted_weights_df["weight"] < top]
    middle_weights = middle_weights[middle_weights["weight"] > bottom]
    
    middle_ids = list(middle_weights.index)
    
    sorted_weights_df.to_csv(f"FR_Test/profiles/{num}.csv")
    
    return middle_ids

def add_inputs_to_file(period, docs, filepath, tokenizer, vectorizer, num):
    tokenized_period = tokenize(docs, "sentence", 1, tokenizer)

    with open(filepath, "w") as train_docs_file:
        train_docs_file.write("doc;mask;period")

    profile = generate_profile(tokenized_period, vectorizer, num)
    with open(filepath, "a") as file:
        for doc in tokenized_period:
            mask = []
            for id in doc:
                mask.append(1 if id in profile else 0)
            file.write(f"\n{doc};{mask};{period}")

    
def tokenize(dataset, mode, size, model):
    chunked_docs = []
    
    for i, row in dataset.iterrows():
        chunked_docs.append(chunk_file(row["filepath"], mode, size))
    
    tokenized_chunks = []
    
    for doc in chunked_docs:
        for chunk in doc:
            
            words = chunk.split()
            
            masked_chunk = []
            
            for word in words:
                if re.match(r"\w+", word):
                    masked_chunk.append(word)
                else:
                    masked_chunk.append("[UNK]")
            
            chunk = " ".join(masked_chunk)
            
            tokenized_chunk = model.encode(chunk)
            tokenized_chunks.append(tokenized_chunk)

    return tokenized_chunks


def separate_periods(df):
    periods = {}
    docs_df = df.sort_values(by="year")
    print(docs_df)
    for index, row in tqdm(docs_df.iterrows(), total=len(docs_df.index), desc="Separating Documents"):
        period = int(PERIOD_LENGTH * np.floor(row["year"]/PERIOD_LENGTH))
        if not period in periods:
            periods[period] = pd.DataFrame(columns=["filepath", "title", "year"])
        r = pd.DataFrame(row).transpose()
        periods[period].loc[len(periods[period])] = r.iloc[0]

    return periods


def get_accuracy(estimate, expected):
    """
    RETURNS TUPLE:
    (`acc`, `acc@3`, `acc@5`)
    """
    #TODO F1 score
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
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.svm import LinearSVC
        from transformers import BertTokenizerFast, BertModel
        from transformers.utils import logging
        import torch
        import multiprocessing as mp
        import copy
        import time
        import re
        import os
        from chunker import chunk_file
        completed = True
    except OSError as e:
        print(e)
        time.sleep(10)

logging.set_verbosity_error()

if __name__ == "__main__":
    print("\rDONE!")

    data_df = pd.read_csv("Documents/_doc_data.csv")

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", dev)

    train, test = train_test_split(data_df, train_size=0.1)

    tfidf_vec = TfidfVectorizer()
    count_vec = CountVectorizer()


    bert_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased").to(dev)

    processes = []

    files = os.listdir("FR_Test/train_docs")
    for file in files:
        file_path = os.path.join("FR_Test/train_docs", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    i=0
    for key, value in tqdm(separate_periods(train).items(), desc="Generating Profiles"):
        process = mp.Process(target=add_inputs_to_file, args=(key, value, f"FR_Test/train_docs/train_docs_{i}.csv", bert_tokenizer, tfidf_vec, i))
        processes.append(process)
        process.start()
        i+=1
    
    train_docs = pd.DataFrame(columns=["doc", "mask", "period"])
    
    for index in tqdm(range(len(processes)), desc="Joining Processes"):
        processes[index].join()
        df = pd.read_csv(f"FR_Test/train_docs/train_docs_{index}.csv", sep=";")
        train_docs = pd.concat([train_docs, df], ignore_index=True)
    
    del processes
        
    def match_lengths(col):
        series = train_docs.loc[:, col]
        max = 0
        for i in series:
            if len(i) > max: max = len(i)
        
        for i in tqdm(range(len(series)), leave=False):
            ls = eval(train_docs.at[i, col])
            train_docs.at[i, col] = ls[:512] + [0]*(512-len(ls))

    match_lengths("doc")
    match_lengths("mask")
    
    train_docs = train_docs.sample(frac=1).reset_index(drop=True)

    svm = LinearSVC()

    BATCH_SIZE = 128
    
    with open("FR_Test/train_outputs.csv", "w") as train_outputs:
        train_outputs.write("output;period")
    
    with open("FR_Test/train_outputs.csv", "a") as train_outputs:

        NUM_BATCHES_TRAIN  = int(np.ceil(len(train_docs.index)/BATCH_SIZE))
        for batch in tqdm(range(NUM_BATCHES_TRAIN)):
            with torch.no_grad():
                first = np.floor(BATCH_SIZE * i)
                last = np.floor(BATCH_SIZE * (i+1))
                
                docs = torch.tensor(train_docs.loc[first:last, "doc"].tolist(), device=dev)
                masks = torch.tensor(train_docs.loc[first:last, "mask"].tolist(), device=dev)
                
                # print(docs)
                # print(masks)
                
                output = bert_model.forward(input_ids=docs, attention_mask=masks).pooler_output.tolist()
                # print(output)
                print(len(output == BATCH_SIZE))
                
                for i in range(BATCH_SIZE):
                    train_outputs.write(f"\n{output[i]};{train_docs.loc[first+i, "period"]}")
    
    train_outputs_df = pd.read_csv("FR_Test/train_outputs.csv", sep=";")
    
    svm.fit(train_outputs_df.loc[:, "output"], train_outputs_df.loc[:, "period"])
    
    tokenized_test = tokenize(test, "sentence", 1, bert_tokenizer)
    test_years = test.loc[:, "year"].tolist()
    print(tokenized_test)
    print(test_years)
    
    test_outputs = []
    
    with open("FR_Test/test_outputs.csv", "w") as test_outputs:
        test_outputs.write("output;period")
    
    with open("FR_Test/test_outputs.csv", "a") as test_outputs:
    
        NUM_BATCHES_TEST  = int(np.ceil(len(test.index)/BATCH_SIZE))
        for i in tqdm(range(NUM_BATCHES_TEST)):
            with torch.no_grad():
                first = np.floor(BATCH_SIZE * i)
                last = np.floor(BATCH_SIZE * (i+1))
                
                docs = torch.tensor(tokenized_test[first:last], device=dev)
                
                output = bert_model.forward(input_ids=docs).pooler_output.tolist()
                
                for i in range(BATCH_SIZE):
                    test_outputs.write(f"\n{output[i]};{test_years[first+i]}")
    
    test_outputs_df = pd.read_csv("FR_Test/test_outputs.csv", sep=";")
        
    estimates = svm.predict(test_outputs_df.loc[:, "output"])
    acc, acc_3, acc_5 = []
    for i in tqdm(range(len(estimates))):
        estimate = estimates.item(i)
        expected = test_years[i]
        accs = get_accuracy(estimate, expected)
        acc.append(accs[0])
        acc_3.append(accs[1])
        acc_5.append(accs[2])
    
    print("Acc:", np.mean(acc))
    print("Acc@3:", np.mean(acc_3))
    print("Acc@5:", np.mean(acc_5))