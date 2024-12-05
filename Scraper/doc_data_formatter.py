import os
import re

documents_dir = "Documents"

os.chdir(documents_dir)

documents = os.listdir()

for doc in documents:
    doc_text = open(doc).read()
    doc_title_split = doc.split("%")