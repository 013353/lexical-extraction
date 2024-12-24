print("Importing packages... ", end="")

import math
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizerFast
import pandas as pd
from chunker import chunk_file
from tqdm import tqdm

print("DONE!")

data_df = pd.read_csv("Documents/_doc_data.csv")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

train, test = train_test_split(data_df, test_size=0.8)

vectorizer = TfidfVectorizer()

def tf_idf(corpus):
    string_corpus = []
    for doc in corpus:
        doc_string = ""
        for token in doc:
            doc_string += str(token) if len(doc_string) == 0 else " " + str(token)
        string_corpus.append(doc_string)
    
    tfidf_vector = vectorizer.fit_transform(string_corpus)
    
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names_out())
    
    token_df = tfidf_df.transpose()
    
    weights = {}
    for token, data in tqdm(token_df.iterrows(), total=len(token_df.index), desc="Averaging"):
        weights[token] = mean(data)
    
    print(weights)
    
    x = []
    y = []
    
    for key, value in weights.items():
        x.append(key)
        y.append(value)

    plt.plot(x, y, "o")
    plt.show()
    
def tokenize(dataset, mode, size, model):
    chunks = []
    
    for i, row in tqdm(dataset.iterrows(), total=len(dataset.index), desc="Chunking"):
        chunks.append(chunk_file(row["filepath"], mode, size))
    
    tokenized_chunks = []
    
    for doc in tqdm(chunks, desc="Tokenizing"):
        for chunk in doc:
            tokenized_chunk = model.encode(chunk)
            tokenized_chunks.append(tokenized_chunk)

    return tokenized_chunks

train_tokens = test_tokens = []

bert = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

for group, dataset in [(train_tokens, train), (test_tokens, test)]:
    group += tokenize(dataset, "sentence", 1, bert)
    
tf_idf(train_tokens)
    
# print(train_tokens)

# train_chunks = []

# for i, row in tqdm(train.iterrows(), total=len(train.index), desc="Chunking"):
#     train_chunks.append(chunk_file(row["filepath"], "word", 200))
    
# tokenized_chunks = []

# bert = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

# for doc in tqdm(train_chunks, desc="Tokenizing"):
#     for chunk in doc:
#         tokenized_chunk = bert.encode(chunk)
#         tokenized_chunks.append(tokenized_chunk)