PERIOD_LENGTH = 10

def generate_profile(corpus, vectorizer, num):
    string_corpus = []
    for doc in corpus:
        doc_string = ""
        for token in doc:
            doc_string += str(token) if len(doc_string) == 0 else " " + str(token)
        string_corpus.append(doc_string)
    
    print(num)
        
    completed = False
    while not completed:
            try:
                arr = vectorizer.fit_transform(string_corpus)
                completed = True
            except MemoryError:
                time.sleep(10)
    # print(arr)
    # input()
    # print(type(arr))
    token_df = pd.DataFrame.sparse.from_spmatrix(arr, columns=vectorizer.get_feature_names_out())
    print(num)
    # token_df = pd.DataFrame(arr, columns=vectorizer.get_feature_names_out())
    
    # for i in range(arr.shape[0]):
    #     # print(token_df)
    #     # print(pd.DataFrame(arr.getrow(i).toarray()).loc[0])
    #     completed = False
    #     while not completed:
    #         try:
    #             token_df.loc[len(token_df.index)] = pd.DataFrame(arr.getrow(i).toarray()).loc[0]
    #             completed = True
    #         except MemoryError:
    #             time.sleep(10)
    # token_df.replace(float("nan"), 0, inplace=True)
    
    # print(token_df.iloc[0,0], "YES")
    
    # token_df.to_csv(f"{__name__}{num}.csv")
    print(num)
    
    print(token_df)
    
    weights = {}
    for index in token_df.index.to_list():
        weights[index] = np.mean(token_df.loc[index])
    
    print(weights)
        
    # for token, data in token_df.iterrows():
    #     weights[token] = np.mean(data)
    # completed = True
    
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=["weight"])
    
    sorted_weights_df = weights_df.sort_values(by=weights_df.columns[0], axis=0, ascending=False)
    
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
    
    # for i, row in sorted_weights_df.iterrows():
    #     print(tokenizer.convert_ids_to_tokens(i), row["weight"])
        
    bottom = sorted_weights_df.quantile(0.125)["weight"]
    top = sorted_weights_df.quantile(0.875)["weight"]
    
    middle_weights = sorted_weights_df[sorted_weights_df["weight"] < top]
    middle_weights = middle_weights[middle_weights["weight"] > bottom]
    
    middle_ids = list(middle_weights.index)
    
    # print("PROFILE GENERATED")
    
    return middle_ids

def add_inputs_to_df(period, docs, dataframe, tokenizer, vectorizer, num):
    tokenized_period = tokenize(docs, "sentence", 1, tokenizer)

    profile = generate_profile(tokenized_period, vectorizer, num)

    for doc in tokenized_period:
        mask = []
        for id in doc:
            mask.append(1 if id in profile else 0)
        # print(mask)
        new_row = [doc, mask, period]
        dataframe.loc[len(dataframe)] = new_row
    
    print("PROCESS DONE")
    
    # print(bert_tokenizer.convert_ids_to_tokens(profile))
    
    # print(period)
    
    # print(dataframe)

    
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
        from sklearn.svm import SVC
        from transformers import BertTokenizerFast, BertModel
        from transformers.utils import logging
        import torch
        import multiprocessing as mp
        # import scipy.sparse
        import copy
        import time
        import re
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
    # bert_tokenizer.to(dev)
    bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased").to(dev)

    train_docs = pd.DataFrame(columns=["doc", "mask", "period"])

    print(train_docs)

    processes = []

    i=2
    
    for key, value in tqdm(separate_periods(train).items(), desc="Generating Profiles"):
        process = mp.Process(target=add_inputs_to_df, args=(key, value, train_docs, bert_tokenizer, tfidf_vec, i))
        process.daemon = True
        processes.append(process)
        process.start()
        i+=1
        # process.join()
        
    for proc in processes:
        proc.join()
        
    def match_lengths(col):
        series = train_docs.loc[:, col]
        max = 0
        for i in series:
            if len(i) > max: max = len(i)
        
        for i in range(len(series)):
            train_docs.at[i, col] = train_docs.at[i, col][:512] + [0]*(512-len(train_docs.at[i, col]))

    match_lengths("doc")
    match_lengths("mask")

    outputs = pd.DataFrame(columns=["output", "period"])

    train_docs = train_docs.sample(frac=1).reset_index(drop=True)
    print(train_docs)

    BATCH_SIZE = 32
    NUM_BATCHES  = np.ceil(len(train_docs.index)/BATCH_SIZE)
    for i in range(NUM_BATCHES):
        first = np.floor(BATCH_SIZE * i)
        last = np.floor(BATCH_SIZE * (i+1))
        
        docs = torch.tensor(train_docs.loc[first:last, "doc"], device=dev)
        masks = torch.tensor(train_docs.loc[first:last, "mask"], device=dev)
        
        print(docs)
        print(masks)
        input()
        
        bert_model.forward(input=docs, attention_mask=masks)
        #TODO

    print(bert_model.forward(input_ids=docs, attention_mask=masks))
    input()
        
    # TODO feed each weighted document into model, "graph", run SVM



    for key, value in tqdm(test.iterrows(), total=len(test.index), desc="Evaluating Test Data"):
        # TODO tokenize, feed into model, "graph", feed into SVM
        pass