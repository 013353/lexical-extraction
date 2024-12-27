print("Importing packages... ", end="")

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
print("Platform:", dev)

train, test = train_test_split(data_df, test_size=0.2)

vectorizer = TfidfVectorizer()

def inverse_tf_idf(corpus):
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
        weights[token] = np.mean(data)
    
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=["weight"])
    
    sorted_weights_df = weights_df.sort_values(by=weights_df.columns[0], axis=0, ascending=False)
    
    # print(sorted_weights_df)
    
    # print(list(sorted_weights_df.index.values)[:5])
    # print(bert.convert_ids_to_tokens(list(sorted_weights_df.index.values)[:5]))
    
    weights_x = []
    
    for i, row in sorted_weights_df.iterrows():
        weights_x.append(row["weight"])
    
    weights_mean = np.mean(range(len(weights_x)))
    weights_std = np.std(range(len(weights_x)))
    
    # print("STDEV:", weights_std)
    # print("MEAN:", weights_mean)
    
    # figure, axis = plt.subplots(2, 1)
    
    # x = []
    # y = []
    # for i, row in sorted_weights_df.iterrows():
    #     x.append(i)
    #     y.append(row["weight"])
        
    # axis[0].plot(x, y)
    
    def bell_curve(x, std, mean, mult=1):
        return mult/(std * np.sqrt(2 * np.pi)) * np.e**( - (x - mean)**2 / (2 * std**2))
    
    i = 0
    for index, row in sorted_weights_df.iterrows():
        i += 1
        sorted_weights_df.at[index, "weight"] = bell_curve(i, weights_std, weights_mean)**4 * row["weight"]
        
    mult = 1/min(list(sorted_weights_df["weight"]))
    
    for index, row in sorted_weights_df.iterrows():
        sorted_weights_df.at[index, "weight"] *= mult
    
    # print(min(list(sorted_weights_df["weight"])))
        
    # x2 = range(sorted_weights_df.shape[0])
    # y2 = []
    # for i, row in sorted_weights_df.iterrows():
    #     y2.append(row["weight"])
        
    # print(sorted_weights_df)
    
    # axis[1].plot(x2, y2, color="r")
    # plt.show()
    
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
    
inverse_tf_idf(train_tokens)