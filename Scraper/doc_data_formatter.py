import os
import re
from tqdm import tqdm
import pandas as pd
import random
from shutil import copy2
import numpy as np

def choose_documents(parent_dir, child_dir, period_length: int, start: int, end:int, limit=150):
    bar = tqdm(desc="Choosing Documents")

    files = pd.read_csv(f"{parent_dir}/_doc_data.csv").sort_values(by="year")

    periods = {}

    for i, row in files.iterrows():
        bar.update()
        period = int(period_length * np.floor(row["year"]/period_length))
        if start <= period <= end:
            if period in periods:
                periods[period].append(row)
            else:
                periods[period] = [row]

    used_docs = []

    for key, value in periods.items():
        bar.update()
        if len(value) > limit:
            random.shuffle(value)
            used_docs.extend(value[:limit])
        else:
            used_docs.extend(value)

    used_docs = pd.DataFrame(used_docs)
    
    print(used_docs)

    for i, row in used_docs.iterrows():
        bar.update()
        copy2(row["filepath"], f"{child_dir}/{row["title"]} %{row["year"]}.txt")

def add_files_to_csv(dir):
    file = open(f"{dir}/_doc_data.csv", "w")
    file.write("filepath,title,year")
    file.close()

    documents = os.listdir(dir)

    for i in tqdm(range(len(documents)), desc="Adding Files to CSV"):
        doc=documents[i]
        if doc.endswith(".txt"):
            try:
                doc_title_split = re.split(r'[.%]', doc)
                doc_title = doc_title_split[0]
                doc_year = re.search(r"\d{4}", doc_title_split[-2]).group(0)
                with open(f"{dir}/_doc_data.csv", "a", encoding="utf-8") as file:
                    file.write(f"\n\"{dir}/{doc}\",\"{doc_title}\",{doc_year}")
            except Exception as e:
                print(e)
                print(doc)

def format_files(dir):
    documents = os.listdir(dir)         
    for filepath in tqdm(documents, desc="Formatting Files"):
        if filepath.endswith(".txt"):
            formatted_doc = ""
            with open(filepath, "r", encoding="utf-8") as read_file:
                doc_lines = read_file.readlines()
                
                for line in doc_lines:
                    match_str = re.match(r"\s+", line)
                    if type(match_str) is re.Match and line != "\n":
                        formatted_line = line[match_str.end():]
                    else:
                        formatted_line = line
                    formatted_doc += formatted_line
            
            with open(filepath, "w", encoding="utf-8") as write_file:
                write_file.write(formatted_doc)

add_files_to_csv("Documents/All Documents")
choose_documents("Documents/All Documents", "Documents", 10, 1800, 2000, 75)
add_files_to_csv("Documents")