import bs4
import re
import requests
from bs4 import BeautifulSoup
import os
import os.path
import csv
import time

#writing the output to a csv file
def writerows(rows,filename):
    with open(filename,'a',encoding='utf-8') as toWrite:
        writer=csv.writer(toWrite)
        writer.writerows(rows)
        
def getlistings(listingurl):
    headers={'User-Agent': 'Mozilla/5.0 (X11: Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox.52.0'}
    try:
        response=requests.get(listingurl,headers=headers)
    except requests.exceptions.RequestException as e:
        print(e)
        exit()
        
    soup=BeautifulSoup(response.text, "html.parser")
    
    listings=[]
    for rows in soup.find_all("th"):
        name=rows.find("div",class_="name").a.get_text()
        Patent=rows.find_all("tr")[0].get_text()
        listings.append([Patent_no])
    return listings

if __name__ == "__main__":
    filename="Patents.csv"
    if os.path.exists(filename):
        os.remove(filename)
        
    baseurl='https://www.uspto.gov/patents-application-process/search-patents'
    page=1
    parturl='http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&p=1&u=%2Fnetahtml%2FPTO%2Fsearch-bool.html&r=0&f=S&l=50&TERM1=Biomedical&FIELD1=&co1=AND&TERM2='

    while page<4:
        listingurl=baseurl+str(page)+parturl
        listings=getlistings(listingurl)
        writerows(listings,filename)
        time.sleep(3)
        page+=1

if page>1:
    print('listings fetched succesfully')
    