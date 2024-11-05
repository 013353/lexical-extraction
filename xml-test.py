from bs4 import BeautifulSoup
import requests
import re
import time

document_names = ["IBE-1", "IBE-2"]

pdf_urls = []

for document_name in document_names:

    doc = open(f"{document_name}.xml")

    soup = BeautifulSoup(doc, "xml")

    sources = soup.find_all("dc:identifier", string=re.compile(r"\.pdf$"))

    pdf_urls += sources

print("NUMBER OF DOCUMENTS:", len(pdf_urls))

for i in range(len(pdf_urls)):
    start_time = time.time()
    print(i)
    pdf_url = pdf_urls[i]
    response = requests.get(pdf_url.string)
    file_save_path = f"Documents/IBE-{i}.pdf"

    if response.status_code == 200:
        with open(file_save_path, 'wb') as file:
            file.write(response.content)
    else:
        print('Failed to download file')
    
    end_time = time.time()
    print((end_time-start_time)/60, " min")

print("DONE")