from bs4 import BeautifulSoup
import requests
import re
import time
import subprocess
import os
import html2text

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
        source_page = requests.get(source_url).text
        source_soup = BeautifulSoup(source_page, "html.parser")

        section_link_elements = source_soup.find("div", class_="main-panel").find("ul").find_all("li", class_="mb-0_5", recursive=False)
        section_links = []
        for section_link_element in section_link_elements:
            section_links.append(section_link_element.find("a", class_="article-link")["href"])


        for section_link in section_links:
            section_page = requests.get(section_link).text
            section_soup = BeautifulSoup(section_page, "html.parser")
            pages = section_soup.find_all("article", class_="fullview-page")
            h = html2text.HTML2Text()
            h.ignore_links = True
            text = ""
            for page in pages:
                print(page)
                input("Press ENTER to continue:")
                try:
                    text += h.handle(page)
                except TypeError:
                    print("PROBLEM")
                    continue
            print(text)
            input("Press ENTER to continue:")
            

    print(links_list)
    print(len(links_list))