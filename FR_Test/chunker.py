import os
import re

def is_long(para):
    return len(para) > 10 and len(para.split()) > 2

def chunk_words(doc_file, num):
    words = re.split(r"\s+", doc_file.read())
    
    split_words = []
    
    cur_split = ""
    
    for i in range(len(words)):
        if i % num == 0 and len(cur_split) > 0:
            split_words.append(cur_split)
            cur_split = ""
        else:
            cur_split += words[i] + " "
    
    return split_words

def chunk_sentences(doc_file):
    doc = re.sub("\n+", "", doc_file.read())
    
    split_doc = re.split(r"([!\.?](?: |$))", doc)
    
    doc_sentences = []
    
    i = 0
    while i < len(split_doc)-1:
        doc_sentences.append(split_doc[i] + split_doc[i+1])
        i += 2

    return doc_sentences

def chunk_paragraphs(doc_file):
    doc_lines = doc_file.readlines()

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
          
          
def chunk_file(filepath, mode, size):
    if os.path.isfile(filepath):
        with open(filepath, encoding="utf-8") as doc_file:
            match mode:
                case "sentence":
                    return chunk_sentences(doc_file)
                case "paragraph":
                    return chunk_paragraphs(doc_file)
                case "word":
                    return chunk_words(doc_file, size)
        
    else:
        raise Exception("Filepath does not lead to a file.")