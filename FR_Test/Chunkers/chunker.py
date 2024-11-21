import os

def is_paragraph(para):
    return len(para) > 30 and len(para.split()) > 2


def separate_paragraphs(filepath):
    if os.path.isfile(filepath):
        with open(filepath) as doc:
            doc_text = doc.read()
            doc_lines = doc_text.splitlines()

            doc_paragraphs = []

            cur_paragraph = ""
            for line in doc_lines:
                if line == "" and cur_paragraph != "":
                    if is_paragraph(cur_paragraph):
                        doc_paragraphs.append(cur_paragraph)
                    cur_paragraph = ""
                else:
                    if cur_paragraph != "":
                        cur_paragraph += "\n"
                    cur_paragraph += line
                        
    return doc_paragraphs

def chunk(filepath, size=1):
    paragraphs = separate_paragraphs(filepath)
    
    chunk = ""
    counter = 1
    chunks = []
    for para in paragraphs:
        chunk += para
        
        if counter % size == 0:
            chunks.append(chunk)
            chunk = ""

        counter += 1