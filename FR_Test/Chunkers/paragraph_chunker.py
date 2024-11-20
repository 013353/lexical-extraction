import os
import re

document_directory = "Documents"

for document_filename in os.listdir(document_directory):
    document_filepath = os.path.join(document_directory, document_filename)

    if os.path.isfile(document_filepath):
        with open(document_filepath) as doc:
            doc_text = doc.read()
            doc_lines = doc_text.splitlines()

            doc_paragraphs = []

            cur_paragraph = ""
            for line in doc_lines:
                if re.match(r'\s', line):
                    doc_paragraphs.append(cur_paragraph)
                else:
                    cur_paragraph.append("\n" + cur_paragraph)