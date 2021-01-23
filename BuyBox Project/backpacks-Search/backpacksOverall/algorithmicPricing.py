#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aayushi

Preprocesses the web scraped data of the BACKPACK dataset
"""

from os import listdir
from os.path import isfile, join
import os
import numpy as np
import itertools

# Preprocesses the buybox and info folder present inside each time stamp folder (obtained by web scraping) to form the feature data folder
def features(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    am = 0
    for folder in folders:
        #print(onlyfiles)
        if not os.path.exists(folder+"/features"):
            os.makedirs(folder+"/features")
        onlyfiles = [f for f in listdir(folder+'/buybox/') if isfile(join(folder+'/buybox/', f))]
        for i in onlyfiles:
            #print(i)
            if len(i.split('.'))>2:
                continue
            featureFile = open(folder+'/features/' + i, 'w')
            sourceFile = open(folder+'/buybox/' + i, 'r')
            infoFile = open(folder+'/info/' + i, 'r')
            featureFile.write('Seller' + '\t' + 'Price' + '\t' + 'No. of Ratings' + '\t' + 'Delivery' + '\t' + 'Price difference to the lowest' + '\t' + 'Price ratio to the lowest' + '\t' + 'Average rating of the seller' + '\t' + 'Positive feedback percentage of the seller' + '\t' + 'Fulfilled by Amazon?' + '\t' + 'Is Amazon the seller?' + '\t' + 'Does is win the buy box?' + '\t' + 'Time' + '\n')
            prices = []
            sellers = []
            noRatings = []
            delivery = []
            avgRating = []
            posFeed = []
            fullfilled = []
            sellerAmazon = []
            a = infoFile.read()
            #print(folder + " " + i)
            winner = ""
            if len(a.split('\t')[8].split(' by '))>1:
                winner = a.split('\t')[8].split(' by ')[1].split(' and ')[0].replace('.', '')
            else:
                winner = a.split('\t')[5]
            winBuybox = []
            b = sourceFile.read().split('\n')[1: -1]
            #print(b)
            for j in b:
                k = j.split('\t')
                #print(k)
                prices.append(k[0].split(' | ')[0].strip().replace(',', ''))
                l = k[2].split(' | ')
                seller_name = l[0].split('(')[0].strip().replace('.', '')
                sellers.append(seller_name)
                try:
                    delivery.append(k[3].split(' | ')[1])
                except:
                    if winner==seller_name:
                        delivery.append(a.split('\t')[2])
                    else:
                        delivery.append('')
                try:
                    if l[3].split()[0].isnumeric():
                        noRatings.append(l[3].split()[0])
                    else:
                        noRatings.append('')
                    avgRating.append(l[1].split()[0])
                    posFeed.append(l[2].split()[0])
                except:
                    noRatings.append('')
                    avgRating.append('')
                    posFeed.append('')
                if k[3].split(' | ')[0]=='Fullfilled by Amazon':
                    fullfilled.append('1')
                else:
                    fullfilled.append('0')
                #print(l[0].split('(')[0].strip())
                if seller_name.lower()=='amazon' or seller_name.lower()=='appario retail private ltd' or seller_name.lower()=='cloudtail india':
                    sellerAmazon.append('1')
                    am = am + 1
                else:
                    sellerAmazon.append('0')
                if seller_name==winner or (winner=='Amazon' and seller_name.lower()=='appario retail private ltd') or (winner=='Amazon' and seller_name.lower()=='cloudtail india'):
                    winBuybox.append('1')
                else:
                    winBuybox.append('0') 
            data = [[sellers[i], prices[i], noRatings[i], delivery[i], avgRating[i], posFeed[i], fullfilled[i], sellerAmazon[i], winBuybox[i]] for i in range(len(b))]
            new_data = list(data for data,_ in itertools.groupby(data))
            
            sellers = [i[0] for i in new_data]
            prices = [i[1] for i in new_data]
            noRatings = [i[2] for i in new_data]
            delivery = [i[3] for i in new_data]
            avgRating = [i[4] for i in new_data]
            posFeed = [i[5] for i in new_data]
            fullfilled = [i[6] for i in new_data]
            sellerAmazon = [i[7] for i in new_data]
            winBuybox = [i[8] for i in new_data]
            minPrice = min([float(price) for price in prices])
            priceDiff = [str(round(abs(float(price) - float(minPrice)), 2)) for price in prices]
            priceRatio = [str(round(float(price)/float(minPrice), 4)) for price in prices]
            
            for j in range(len(new_data)):
                featureFile.write(sellers[j] + '\t' + prices[j] + '\t' + noRatings[j] + '\t' + delivery[j] + '\t' + priceDiff[j] + '\t' + priceRatio[j] + '\t' + avgRating[j] + '\t' + posFeed[j] + '\t' + fullfilled[j] + '\t' + sellerAmazon[j] + '\t' + winBuybox[j] + '\t' + folder + '\n')
            featureFile.close()
            sourceFile.close()
            infoFile.close()
    #print(am)

def main():
    # Preprocessing the web scraped data data
    # Change the argument to the path with the backpacks-Search/backpacksOverall folder
	features(os.getcwd())

if __name__=='__main__':
	main()

