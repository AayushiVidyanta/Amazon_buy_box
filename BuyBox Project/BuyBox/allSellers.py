#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aayushi

Preprocesses the web scraped data of the HEADPHONE dataset to find all the sellers
"""

from os import listdir
from os.path import isfile, join
from collections import defaultdict

# Preprocesses the buybox and info folder present (obtained by web scraping) to find all the sellers in the dataset
def sellers(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    sellersFile = open('sellers.tsv', 'w')
    seller = defaultdict(lambda: [0, 0])
    for i in onlyfiles:
        if len(i.split('.'))>2:
            continue
        buyboxFile = open(mypath + '/' + i, 'r')
        infoFile = open('./info/' + i, 'r')
        a = buyboxFile.read().split('\n')[1: -1]
        b = infoFile.read()
        for j in a:
            k = j.split('\t')
            l = k[2].split(' | ')
            seller_name = l[0].split('(')[0].strip()
            seller_name = seller_name.replace('.', '')
            seller[seller_name][0]+=1
        winner = b.split('\t')[8].split(' (')[0].split(' by ')[-1].replace('.', '')
        seller[winner][1]+=1
        
    sellersFile.write('Seller' + '\t' + 'No. of times it competes for buybox' + '\t' + 'No. of times it wins the buybox' + '\n')
    for item in seller.items():
        #assert(item[1][0]>item[1][1])
        sellersFile.write(item[0] + '\t' + str(item[1][0]) + '\t' + str(item[1][1]) + '\n')
    sellersFile.close()

def main():
    # Preprocessing the web scraped data data to find all the sellers in the dataset
    # Change the argument to the path with the buybox folder
	sellers('./buybox')

if __name__=='__main__':
	main()
