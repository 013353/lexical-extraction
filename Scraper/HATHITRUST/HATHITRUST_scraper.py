from bs4 import BeautifulSoup
import requests
import re
import time
import subprocess
import os
import scraper_tools

num_pages = 24

for result_page_num in range(1, num_pages + 1):
    start_time = time.time()
    search_results_page_url = f"https://catalog.hathitrust.org/Search/Home?type%5B%5D=all&lookfor%5B%5D=%2A&filter%5B%5D=topicStr%3AHistory&filter%5B%5D=language%3AEnglish&sort=yearup&page={result_page_num}&pagesize=100&ft=ft"
    search_results_page = requests.get(search_results_page_url).text
    
    search_results = BeautifulSoup(search_results_page, "html.parser")\
                     .find_all("article", class_="record d-flex gap-3 p-3 mb-3 mt-3 shadow-sm")
    
    for result in search_results:
        try:
            result_button = result.find("a")
            
            result_catalog_url = "https://catalog.hathitrust.org" + result_button["href"]
            
            result_text_url = "https:" + \
                BeautifulSoup(requests.get(result_catalog_url).text, "html.parser")\
                .find("span", string="Full view")\
                .parent\
                ["href"]\
                .replace("pt?", "ssd?")
        except Exception as e:
            print(e)
            continue
        
        doc_text = ""
        breakout = False
        page_num = 0
        
        while not breakout:
            page_num += 1
            
            page_soup = BeautifulSoup(requests.get(result_text_url + "&seq=" + str(page_num)).text, "html.parser")
            
            if page_num == 1:
                doc_title = page_soup.find("span", attrs={"property": "dc:title"}).string
                doc_year = re.search(r"\b\d{4}", page_soup.find("span", attrs={"property": "dc:publisher"}).string).group(0)
            
            # with open("log.txt", 'a') as file:
            #     file.write("----------------------------------------------------------------------------------------------\n" + page_soup.prettify())
            
            print(page_soup.find("p", string=re.compile("This is the last page")))
            
            paragraphs = page_soup.find_all("p")
            
            last_page = False
            
            for para in paragraphs:
                if (para.strings != []):
                    for string in para.strings:
                        if re.search(re.compile("This is the last page"), string):
                            last_page = True
        
            if last_page:
                print("last page")
                breakout = True
                break
            elif page_soup.find("div", id="mdpTextEmpty"):
                print("page skipped")
                continue
            
            # print(doc_text)
            page_text = page_soup.find("p", class_="Text").contents[0] + "\n"
            doc_text += page_text
            print(page_text)
            print("page added", page_num)
        print(doc_text)
        
        doc_file = open(f"Documents/{scraper_tools.format_name(doc_title)} %{doc_year}.txt", "w")
        doc_file.write(doc_text)
        
        print(time.time()-start_time)
        # print(scraper_tools.colors.HEADER, f"{counter}/{len(documents)}", scraper_tools.colors.ENDC, time.time()-start_time,  "sec")