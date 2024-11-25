from bs4 import BeautifulSoup
import requests
import re
import time
import subprocess
import os
# import scraper_tools

num_pages = 24

for page_num in range(1, num_pages + 1):
    search_results_page_url = f"https://catalog.hathitrust.org/Search/Home?type%5B%5D=all&lookfor%5B%5D=%2A&filter%5B%5D=topicStr%3AHistory&filter%5B%5D=language%3AEnglish&sort=yearup&page={page_num}&pagesize=100&ft=ft"
    search_results_page = requests.get(search_results_page_url).text
    
    search_results = BeautifulSoup(search_results_page, "html.parser")\
                     .find_all("article", class_="record d-flex gap-3 p-3 mb-3 mt-3 shadow-sm")
    
    for result in search_results:
        result_button = result.find("a")
        
        result_catalog_url = "https://catalog.hathitrust.org" + result_button["href"]
        
        result_text_url = "https:" + \
            BeautifulSoup(requests.get(result_catalog_url).text, "html.parser")\
            .find("span", string="Full view")\
            .parent\
            ["href"]\
            .replace("pt?", "ssd?")
            
        print(result_text_url)