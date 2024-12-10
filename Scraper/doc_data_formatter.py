import os
import re
from tqdm import tqdm

documents_dir = "Documents"

os.chdir(documents_dir)

documents = os.listdir()

file = open("_doc_data.csv", "w")
file.write("filepath,title,year")

for i in tqdm(range(len(documents)), 
               desc="Loading…", 
               ascii=False, ncols=75):
    doc=documents[i]
    if doc.endswith(".txt"):
        try:
            doc_text = open(doc).read()
            doc_title_split = re.split(r'[.%]', doc)
            doc_title = doc_title_split[0]
            doc_year = re.search(r"\d{4}", doc_title_split[-2]).group(0)
            with open("_doc_data.csv", "a") as file:
                file.write(f"\n\"Documents/{doc}\",\"{doc_title}\",{doc_year}")
        except Exception as e:
            print(e)
            print(doc)