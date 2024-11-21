import bs4
import requests
import time
import scraper_tools

search_results_page = open(f"Scraper/IA/IA.xml")

search_results_soup = bs4.BeautifulSoup(search_results_page, "xml")

documents = search_results_soup.find_all("doc")

print(scraper_tools.scraper_tools.colors.OKCYAN, "NUMBER OF DOCUMENTS:", len(documents), scraper_tools.colors.ENDC)

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

        text_file_path = f"Documents/{scraper_tools.format_name(doc_title)} %{doc_year}.txt"

        # [response code]          OK    
        if response.status_code == 200:
            with open(text_file_path, 'wb') as file:
                file.write(response.content)
                print(scraper_tools.colors.HEADER, f"{counter}/{len(documents)}", scraper_tools.colors.ENDC, time.time()-start_time,  "sec")

        else:
            print(scraper_tools.colors.HEADER, f"{counter}/{len(documents)}", scraper_tools.colors.FAIL, "ERROR", response.status_code, scraper_tools.colors.ENDC)

    else:
        print(scraper_tools.colors.HEADER, f"{counter}/{len(documents)}", scraper_tools.colors.FAIL, "INSUFFICIENT DATA", scraper_tools.colors.ENDC)

    counter += 1

print(scraper_tools.colors.OKGREEN, "DONE", scraper_tools.colors.ENDC)