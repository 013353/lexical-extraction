import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import pandas as pd
from chunker import chunk_file
from tqdm import tqdm

data_df = pd.read_csv("Documents/_doc_data.csv")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

train, test = train_test_split(data_df, test_size=0.2)

chunks = []

for i, row in tqdm(train.iterrows(), total=len(train.index), desc="Chunking"):
    chunks.append(chunk_file(row["filepath"]))
    
tokens = []

bert = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

for doc in tqdm(chunks, desc="Tokenizing"):
    tokenized_doc = []
    for chunk in doc:
        
        
        
        tokenized_chunk = bert.encode(chunk)
        # print(tokenized_chunk)
        # input()
        tokenized_doc.append(tokenized_chunk)