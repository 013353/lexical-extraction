import os
import re
from tqdm import tqdm
import pandas as pd
import random
import matplotlib.pyplot as plt
from shutil import copy2

def choose_documents(start, end):
    bar = tqdm(desc="Choosing Documents")

    files = pd.read_csv(f"{start}/_doc_data.csv").sort_values(by="year")

    years = {}

    for i, row in files.iterrows():
        bar.update()
        period = int(str(row["year"])[:3] + "0")
        if period in years:
            years[period].append(row)
        else:
            years[period] = [row]

    limit = 150

    used_docs = []

    for key, value in years.items():
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
        copy2(row["filepath"], f"{end}/{row["title"]} %{row["year"]}.txt")

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
                
add_files_to_csv("Documents")