import bs4
import requests
import re
import time

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

search_results_page = open(f"Scraper/IA/IA.xml")

search_results_soup = bs4.BeautifulSoup(search_results_page, "xml")

documents = search_results_soup.find_all("doc")

print(colors.OKCYAN, "NUMBER OF DOCUMENTS:", len(documents), colors.ENDC)

counter = 1
for doc in documents:
    start_time = time.time()

    doc_id = doc.find("str", attrs={"name": "identifier"})
    doc_title = doc.find("str", attrs={"name": "title"})
    doc_year = doc.find("str", attrs={"name": "year"})

    if doc_id and doc_title and doc_year:

        doc_id = doc_id.string
        doc_title = doc_title.string
        doc_year = doc_year.string

        doc_url = f"https://archive.org/download/{doc_id}/{doc_id}_djvu.txt"
        response = requests.get(doc_url)

        text_file_path = f"Documents/{doc_title} %{doc_year}.txt"

        # [response code]          OK    
        if response.status_code == 200:
            with open(text_file_path, 'wb') as file:
                file.write(response.content)
                print(colors.HEADER, f"{counter}/{len(documents)}", colors.ENDC, time.time()-start_time,  "sec")

        else:
            print(colors.HEADER, f"{counter}/{len(documents)}", colors.FAIL, "ERROR", response.status_code, colors.ENDC)

    else:
        print(colors.HEADER, f"{counter}/{len(documents)}", colors.FAIL, "INSUFFICIENT DATA", colors.ENDC)

    counter += 1

print(colors.OKGREEN, "DONE", colors.ENDC)