from bs4 import BeautifulSoup
import requests
import re

document_names = ["IBE-1", "IBE-2"]

pdf_urls = []

for document_name in document_names:

    doc = open(f"{document_name}.xml")

    soup = BeautifulSoup(doc, "xml")

    sources = soup.find_all("dc:identifier", string=re.compile(r"\.pdf$"))

    pdf_urls += sources

print(len(pdf_urls))


for pdf_url in pdf_urls:
    response = requests.get(pdf_url)
    file_save_path = f"Documents/{document_name}.pdf"

    if response.status_code == 200:
        with open(file_save_path, 'wb') as file:
            file.write(response.content)
    else:
        print('Failed to download file')