from bs4 import BeautifulSoup
import requests
import re
import time

file_names = ["IBEUNESCO-1", "IBEUNESCO-2", "IBEUNESCO-3"]

documents = {}


for file in file_names:

    doc = open(f"Scraper/IBEUNESCO/{file}.xml")

    soup = BeautifulSoup(doc, "xml")

    sources_list = soup.find_all("reference")

    for source in sources_list:
        try:
            all_urls = source.find("ul", string=re.compile(r"\.pdf")).string.split(";\n\t")
        except AttributeError:
            continue

        document_name = source.find("t1").string

        document_year = source.find("yr").string

        for url in all_urls:
            if url.endswith(".pdf"):
                documents.setdefault(document_name, (document_year, url))


print("NUMBER OF DOCUMENTS:", len(documents))

for name, info in documents.items():
    start_time = time.time()

    pdf_url = info[1]
    document_year = info[0]

    response = requests.get(pdf_url)

    file_save_path = f"Documents/{name} %{document_year}.pdf"

    if response.status_code == 200:
        with open(file_save_path, 'wb') as file:
            file.write(response.content)
    else:
        print('Failed to download file')
    
    end_time = time.time()
    print(end_time-start_time, "sec")

print("DONE")