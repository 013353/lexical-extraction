import bs4
import requests
import re
import time

sources = ["ECCO", "EEBO", "EVANS"]

for source in sources:

    search_results_page = open(f"Scraper/TCP/TCP-{source}.html")

    search_results_soup = bs4.BeautifulSoup(search_results_page, "html.parser")

    document_links = search_results_soup.find_all("a", class_="results-link")

    print('\033[96m', "NUMBER OF DOCUMENTS:", len(document_links), '\033[0m')

    counter = 1

    for doc in document_links:
        start_time = time.time()
        
        print('\033[95m', f"{counter}/{len(document_links)}", '\033[0m', end=": ")
        
        doc_url = doc["href"]
        doc_page = requests.get(doc_url).text
        doc_soup = bs4.BeautifulSoup(doc_page, "html.parser")
        
        # [regex]          either . or / 
        doc_title = re.sub("[./]", "", doc_soup.find("div", attrs={"data-key": "title"}).find("dd").contents[0])
        
        pubinfo = doc_soup.find("div", attrs={"data-key": "pubinfo"}).find_all("dd")
        for line in pubinfo:
            # [regex]                exactly four digits at a word boundary
            year_search = re.search(r"\b\d{4}", str(line))
            if year_search:
                doc_year = year_search.group(0)
        
        doc_text = ""

        section_link_elements = doc_soup.find("div", class_="main-panel").find("ul").find_all("li", class_="mb-0_5", recursive=False)
        section_links = []
        for section_link_element in section_link_elements:
            section_links.append(section_link_element.find("a", class_="article-link")["href"])

        for section_link in section_links:
            section_page = requests.get(section_link).text
            section_soup = bs4.BeautifulSoup(section_page, "html.parser")
            paragraphs = section_soup.find_all("p", attrs={"data-debug": "otherwise"})
            
            for para in paragraphs:
                
                for child in para.contents:
                    if type(child) != bs4.element.NavigableString:
                        if child.name in ["span", "blockquote", "highlight"]:
                            child.unwrap()
                        else:
                            child.decompose()
                    
                para.smooth()
                
                if len(para) > 0:
                    try: doc_text += para.contents[0]
                    except TypeError: para.unwrap
        
        doc_file = open(f"Documents/{doc_title[:150]} %{doc_year}.txt", "w")
        doc_file.write(doc_text)
        
        print(time.time() - start_time, "sec")
        counter += 1
    
print('\033[92m', "DONE", '\033[0m')