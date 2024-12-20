import os
import re
from tqdm import tqdm

documents_dir = "Documents"

os.chdir(documents_dir)

documents = os.listdir()

file = open("_doc_data.csv", "w")
file.write("filepath,title,year")
file.close()

for i in tqdm(range(len(documents)), desc="Adding Files to CSV"):
    doc=documents[i]
    if doc.endswith(".txt"):
        try:
            doc_title_split = re.split(r'[.%]', doc)
            doc_title = doc_title_split[0]
            doc_year = re.search(r"\d{4}", doc_title_split[-2]).group(0)
            with open("_doc_data.csv", "a") as file:
                file.write(f"\n\"Documents/{doc}\",\"{doc_title}\",{doc_year}")
        except Exception as e:
            print(e)
            print(doc)
            
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