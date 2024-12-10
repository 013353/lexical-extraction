import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from chunker import chunk_file

data_df = pd.read_csv("Documents/doc_data.csv")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

train, test = train_test_split(data_df, test_size=0.2)

print(train)

for i, row in train.iterrows():
    print(row["filepath"])
    print(chunk_file(row["filepath"]))