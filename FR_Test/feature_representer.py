print("Importing packages... ")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BertTokenizerFast, BertModel
import torch
# import scipy.sparse
import copy
from chunker import chunk_file

print("\rDONE!")


PERIOD_LENGTH = 10

def generate_profile(corpus, vectorizer, model):
    string_corpus = []
    for doc in corpus:
        doc_string = ""
        for token in doc:
            doc_string += str(token) if len(doc_string) == 0 else " " + str(token)
        string_corpus.append(doc_string)
        
    arr = vectorizer.fit_transform(string_corpus).toarray()
    
    token_df = pd.DataFrame(columns=vectorizer.get_feature_names_out())
    
    for row in arr:
        token_df.loc[len(token_df.index)] = row
    
    
    weights = {}
    for token, data in token_df.iterrows():
        weights[token] = np.mean(data)
    
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
    
    sorted_weights_df = sorted_weights_df.sort_values(by=weights_df.columns[0], axis=0, ascending=False)
    
    for i, row in sorted_weights_df.iterrows():
        print(bert_tokenizer.convert_ids_to_tokens(i), row["weight"])
        
    q1 = sorted_weights_df.quantile(0.25)
    q3 = sorted_weights_df.quantile(0.75)
    
    print(q1, q3)

    return sorted_weights_df
    
def tokenize(dataset, mode, size, model):
    chunks = []
    
    for i, row in dataset.iterrows():
        chunks.append(chunk_file(row["filepath"], mode, size))
    
    tokenized_chunks = []
    
    for doc in chunks:
        for chunk in doc:
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



data_df = pd.read_csv("Documents/_doc_data.csv")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", dev)

train, test = train_test_split(data_df, train_size=0.2)

tfidf_vec = TfidfVectorizer()
count_vec = CountVectorizer()

profiles = {}

bert_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
# bert_tokenizer.to(dev)
bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")

for key, value in tqdm(separate_periods(train).items(), desc="Generating Profiles"):

    train_tokens = tokenize(value, "sentence", 1, bert_tokenizer)

    profiles[key] = generate_profile(train_tokens, tfidf_vec, bert_model)
    
    # TODO apply sorted_weights_df to each document in corpus, return result
    # TODO feed each weighted document into model, "graph", run SVM



for key, value in tqdm(test.iterrows(), total=len(test.index), desc="Evaluating Test Data"):
    # TODO tokenize, feed into model, "graph", feed into SVM
    pass

print(profiles)