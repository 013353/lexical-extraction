import os

def is_paragraph(para):
    return len(para) > 10 and len(para.split()) > 2


def chunk_paragraphs(filepath):
    if os.path.isfile(filepath):
        with open(filepath) as doc:
            doc_lines = doc.readlines()
            print(doc_lines)
            # doc_lines = doc_text.splitlines()

            doc_paragraphs = []

            cur_paragraph = ""
            for line in doc_lines:
                ln = line.replace("\n", "")
                if ln == "" and cur_paragraph != "":
                    if is_paragraph(cur_paragraph):
                        doc_paragraphs.append(cur_paragraph)
                    cur_paragraph = ""
                else:
                    if cur_paragraph != "":
                        cur_paragraph += "\n"
                    cur_paragraph += ln
            if is_paragraph(cur_paragraph):
                doc_paragraphs.append(cur_paragraph)                  
    return doc_paragraphs

def chunk_file(filepath, size=1):
    paragraphs = chunk_paragraphs(filepath)
    
    print(paragraphs)
    
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