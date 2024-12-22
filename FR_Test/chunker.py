import os
import re

def is_long(para):
    return len(para) > 10 and len(para.split()) > 2


def chunk_paragraphs(filepath):
    if os.path.isfile(filepath):
        with open(filepath, encoding="utf-8") as doc:
            doc_lines = doc.readlines()
            # print(doc_lines)
            # doc_lines = doc_text.splitlines()

            doc_paragraphs = []

            cur_paragraph = ""
            for line in doc_lines:
                ln = line.replace("\n", "")
                if ln == "" and cur_paragraph != "":
                    if is_long(cur_paragraph):
                        doc_paragraphs.append(cur_paragraph)
                    cur_paragraph = ""
                else:
                    if cur_paragraph != "":
                        cur_paragraph += "\n"
                    cur_paragraph += ln
            if is_long(cur_paragraph):
                doc_paragraphs.append(cur_paragraph)             
            return doc_paragraphs
    else:
        raise Exception("Filepath does not lead to a file.")

def chunk_sentences(filepath):
    if os.path.isfile(filepath):
        with open(filepath, encoding="utf-8") as doc_file:
            doc = re.sub("\n+", "", doc_file.read())
            
            split_doc = re.split(r"([!\.?](?: |$))", doc)
            
            doc_sentences = []
            
            i = 0
            while i < len(split_doc)-1:
                doc_sentences.append(split_doc[i] + split_doc[i+1])
                i += 2
            print(doc_sentences)
    else:
        raise Exception("Filepath does not lead to a file.")

chunk_sentences("Documents/The philosophy of history, by Georg Wilhelm Friedrich Hegel_ with prefaces by Charles Hegel and the translator, J_ Sibree, M_A_,  _ _1900 (1).txt")
            

def chunk_file(filepath, size=1):
    paragraphs = chunk_paragraphs(filepath)
    
    # print(paragraphs)
    
    chunk = ""
    counter = 1
    chunks = []
    for para in paragraphs:
        chunk += para
        
        if counter % size == 0:
            chunks.append(chunk)
            chunk = ""

        counter += 1
        
    return chunks