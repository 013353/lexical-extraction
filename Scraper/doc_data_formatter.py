import os
import re

documents_dir = "Documents"

os.chdir(documents_dir)

documents = os.listdir()

file = open("doc_data.csv", "w")
file.write("filepath,title,year")

for doc in documents:
    if doc.endswith(".txt"):
        doc_text = open(doc).read()
        doc_title_split = re.split(r'[.%]', doc)
        doc_title = doc_title_split[0]
        doc_year = re.search(r"\d{4}", doc_title_split[-2]).group(0)
        print(doc_title, doc_year)
        with open("doc_data.csv", "a") as file:
            file.write(f"\nDocuments/{doc},{doc_title},{doc_year}")