from bs4 import BeautifulSoup
import requests
import re
import time
import subprocess
import os

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

file_names = ["IBEUNESCO-1", "IBEUNESCO-2", "IBEUNESCO-3"]

documents = {}


for search_file_name in file_names:

    search_file = open(f"Scraper/IBEUNESCO/{search_file_name}.xml")

    soup = BeautifulSoup(search_file, "xml")

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


print(colors.OKCYAN, "NUMBER OF DOCUMENTS:", len(documents), colors.ENDC)

counter = 1

for name, info in documents.items():
    start_time = time.time()

    pdf_url = info[1]
    document_year = info[0]

    response = requests.get(pdf_url)

    pdf_file_path = f"Documents/{name[:150]} %{document_year}.pdf"

    # [response code]          OK
    if response.status_code == 200:
        with open(pdf_file_path, 'wb') as file:
            file.write(response.content)
            text_file_path = f"Documents/{name[:150]} %{document_year}.txt"
            subprocess.run(["ocrmypdf", pdf_file_path, pdf_file_path, "--remove-background"])
            subprocess.run(["ebook-convert", pdf_file_path, text_file_path, "--pdf-engine", "pdftohtml","--enable-heuristics"])
            os.remove(pdf_file_path)
    else:
        print("Failed to download file")
    
    end_time = time.time()
    print(colors.HEADER, f"{counter}/{len(documents)}", colors.ENDC, time.time()-start_time,  "sec")
    counter += 1

print(colors.OKGREEN, "DONE", colors.ENDC)