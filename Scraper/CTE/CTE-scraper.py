from bs4 import BeautifulSoup
import requests
import re
import time
import subprocess
import os

file_names = ["CTE"]

documents = {}


for file in file_names:

    search_results = open(f"Scraper/CTE/{file}.html")

    soup = BeautifulSoup(search_results, "html.parser")

    sources_list = soup.find_all("a", class_="results-link")
    
    links_list = []
    for source in sources_list:
        source_url = source["href"].split("?")[0]
        links_list.append(source_url)
        source_page = open(source_url)
        source_soup = BeautifulSoup(source_page, "html.parser")

        section_links_list = source_soup.find("div", class_ = "main-panel").find("ul").find_all("li", class_="mb-0_5")

        for section_link in section_links_list:
            section_page = open(section_link)

    print(links_list)
    print(len(links_list))

#     for source in sources_list:
#         try:
#             all_urls = source.find("ul", string=re.compile(r"\.pdf")).string.split(";\n\t")
#         except AttributeError:
#             continue

#         document_name = source.find("t1").string

#         document_year = source.find("yr").string

#         for url in all_urls:
#             if url.endswith(".pdf"):
#                 documents.setdefault(document_name, (document_year, url))


# print("NUMBER OF DOCUMENTS:", len(documents))

# counter = 1

# for name, info in documents.items():
#     start_time = time.time()

#     pdf_url = info[1]
#     document_year = info[0]

#     response = requests.get(pdf_url)

#     pdf_file_path = f"Documents/{name} %{document_year}.pdf"

#     if response.status_code == 200:
#         with open(pdf_file_path, 'wb') as file:
#             file.write(response.content)
#             text_file_path = f"Documents/{name} %{document_year}.txt"
#             subprocess.run(["ebook-convert", pdf_file_path, text_file_path, "--enable-heuristics"])
#             os.remove(pdf_file_path)
#     else:
#         print("Failed to download file")
    
#     end_time = time.time()
#     print(f"{counter}/{len(documents)}: {end_time-start_time} sec")
#     counter += 1

# print("DONE")