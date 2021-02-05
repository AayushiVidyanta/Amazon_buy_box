#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:17:34 2020

@author: aayushi

Experiment 1
Considers the sellers of the Amazon sold products on the BACKPACK dataset
Amazon sold products: Products where Cloudtail India or Appario Retail Private Ltd is one of the competing sellers
"""

import pandas as pd
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
import matplotlib.pyplot as plt 
import os
import matplotlib
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# Finds the Amazon sold products from the feature data folder
# Returns amazonSold, fba
# amazonSold is a list. amazonSold[i] is 1 if the ith index product is Amazon sold, else 0
# fba is a list. fba[i] is 1 if at least one competing seller of the ith index product is FBA and amazonSold[i] is 0, else 0 
def amazonSoldProducts(mypath):
    amazonSold = []
    fba = []
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
        
        for i in onlyfiles:
            #print(i)
            if len(i.split('.'))>2:
                continue
            with open(folder+'/features/' + i, 'r') as featureFile:
                read_tsv = csv.reader(featureFile, delimiter='\t')
                j = 0
                flag1 = 0
                flag2 = 0
                for row in read_tsv:
                    if j==0:
                        j = j+1
                        continue
                    if len(row)==12:
                        if len(row[9])>0:
                            row[9] = int(row[9])
                            if row[9]==1:                
                                flag1 = 1
                        if len(row[8])>0:
                            row[8] = int(row[8])
                            if row[8]==1:
                                flag2 = 1
                if flag1:
                    amazonSold.append(1)
                else:
                    amazonSold.append(0)
                if flag2 and not flag1:
                    fba.append(1)
                else:
                    fba.append(0)
                
    return amazonSold, fba

# Preprocesses the feature data folder considering only the sellers of the Amazon sold products 
# Returns features, sellers
# features is a pandas dataframe. Each row corresponds to a offer by a seller for a product.
# Its columns are: 'Product ID', 'Rank', 'Seller', 'Price', 'Feedback Count', 'Delivery', 'Price difference to the lowest', 'Price ratio to the lowest', 'Avg rating', 'Positive feedback', 'Fulfilled by Amazon?', 'Is Amazon the seller?', 'Does it win the buy box?', 'Timestamp', 'Number of Sellers' 
# ================================================================= FEATURES ============================================================================
# 1.  Product ID: Unique product ID given by Amazon to each of the products sold on their website
# 2.  Rank: The position at which the seller appears on the seller page of the particular product
# 3.  Seller: Seller name
# 4.  Price: Price of the product
# 5.  Feedback Count: Number of ratings of the seller
# 6.  Delivery: Estimated date of delivery for the product by the seller
# 7.  Price difference to the lowest: Difference of the price by the seller and the lowest price offered for the product among all competing sellers
# 8.  Price ratio to the lowest: Ratio of the price by the seller to the lowest price offered for the product among all the competing sellers
# 9.  Avg rating: Average rating of the seller
# 10. Positive feedback: Positive feedback percentage of the seller
# 11. Fulfilled by Amazon?: 1 if the seller is fulfilled by amazon (FBA), else 0
# 12. Is Amazon the seller?: 1 if the seller is Cloudtail India or Appario Retail Private Ltd, else 0 
# 13. Does it win the buy box?: 1 if the seller wins the buy box for that product, else 0
# 14. Timestamp: Timestamp of when the seller data is collected
# 15. Number of sellers: Total number of competing sellers for that product
# =======================================================================================================================================================
# sellers is a dictionary. sellers[product ID] is the number of competing sellers for the product with that product ID
def preprocess_amazon_sold_products(mypath, amazonSold):
    data = []
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    sellers = dict()
    row_no = 0
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
        #print(onlyfiles)
        r = l
        for i in onlyfiles:
            #print(i)
            if len(i.split('.'))>2:
                continue
            product_name = i.split('.')[0]
            if amazonSold[r]==1:
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    #print(no_of_sellers)
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        row_new = row.copy()
                        row_new.insert(0, product_name)
                        row_new.insert(1, j)
                        if len(row_new)==14:
                            #print(row_new)
                            if len(row_new[9])>0:
                                row_new[9] = float(row_new[9][:-1])
                            else:
                                row_new[9] = np.nan
                            if len(row_new[4])>0:
                                row_new[4] = int(row_new[4].replace(',', ''))
                            else:
                                row_new[4] = np.nan
                            if len(row_new[3])>0:
                                row_new[3] = float(row_new[3])
                            else:
                                row_new[3] = np.nan
                            if len(row_new[6])>0:
                                row_new[6] = float(row_new[6])
                            else:
                                row_new[6] = np.nan
                            if len(row_new[7])>0:
                                row_new[7] = float(row_new[7])
                            else:
                                row_new[7] = np.nan
                            if len(row_new[8])>0:
                                row_new[8] = float(row_new[8])
                            else:
                                row_new[8] = np.nan
                            if len(row_new[10])>0:
                                row_new[10] = int(row_new[10])
                            else:
                                row_new[10] = np.nan
                            if len(row_new[11])>0:
                                row_new[11] = int(row_new[11])
                            else:
                                row_new[11] = np.nan
                            if len(row_new[12])>0:
                                row_new[12] = int(row_new[12])
                            else:
                                row_new[12] = np.nan
                            #print(row_new)
                            data.append(row_new)
                            j = j+1
            r = r+1

        for i in onlyfiles:
            #print(l)
            if len(i.split('.'))>2:
                continue
            product_name = i.split('.')[0]
            if amazonSold[l]==1:
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    list1 = list(read_tsv)
                    no_of_sellers = len(list1)-1
                    sellers[product_name] = no_of_sellers
                    for j in range(no_of_sellers):
                        data[row_no].append(no_of_sellers)
                        row_no = row_no + 1
            l = l+1
            
    #print(data)
    features = pd.DataFrame(data, columns=['Product ID', 'Rank', 'Seller', 'Price', 'Feedback Count', 'Delivery', 'Price difference to the lowest', 'Price ratio to the lowest', 'Avg rating', 'Positive feedback', 'Fulfilled by Amazon?', 'Is Amazon the seller?', 'Does it win the buy box?', 'Timestamp', 'Number of Sellers'])
    return features, sellers

# Prints the percentage of buy box winners which offer the lowest price
def priceWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if len(row[4])>0:
                                    if float(row[4])==0:
                                        t1 = t1 + 1
                                    else:
                                        t0 = t0 + 1
            l = l+1
    t = t0+t1
    print("Lowest Price")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))  

# Prints the percentage of buy box winners which have the lowest rank
def rankWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if j==1:
                                    t1 = t1 + 1
                                else:
                                    t0 = t0 + 1
                        j = j+1
            l = l+1
    t = t0+t1
    print("Lowest Rank")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))                    

# Prints the percentage of buy box winners which have the highest positive feedback    
def posfbWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                posfbs = []
                #print(folder, i)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[7])>0:
                            posfbs.append(int(row[7].split('%')[0]))
                        else:
                            posfbs.append(0)
                        j = j+1
                maxposfb = max(posfbs)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if posfbs[j-1]==maxposfb:
                                    t1 = t1 + 1
                                else:
                                    t0 = t0 + 1
                        j = j+1
            l = l+1
    t = t0+t1
    print("Highest Postive Feedback")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))
 
# Prints the percentage of buy box winners which have the highest rating count
def ratingCntWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                ratingCnt = []
                #print(folder, i)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[2])>0:
                            ratingCnt.append(int(row[2].split('%')[0]))
                        else:
                            ratingCnt.append(0)
                        j = j+1
                maxRatingCnt = max(ratingCnt)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if ratingCnt[j-1]==maxRatingCnt:
                                    t1 = t1 + 1
                                else:
                                    t0 = t0 + 1
                        j = j+1
            l = l+1
    t = t0+t1
    print("Highest Rating Count")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))

# Prints the percentage of buy box winners which have the highest average rating
def avgRatingWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                avgRat = []
                #print(folder, i)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[6])>0:
                            avgRat.append(float(row[6]))
                        else:
                            avgRat.append(0)
                        j = j+1
                maxAvgRat = max(avgRat)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if avgRat[j-1]==maxAvgRat:
                                    t1 = t1 + 1
                                else:
                                    t0 = t0 + 1
                        j = j+1
            l = l+1
    t = t0+t1
    print("Highest Average Rating")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))

# Helper function for plotting the distribution of buy box winners with ranks 
# Returns ranks
# ranks is a dict. ranks[i] is the number of buy box winners with rank i    
def plotRankWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    ranks = defaultdict(lambda: 0)
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                ranks[j] = ranks[j] + 1
                        j = j+1
            l = l+1
    return ranks             
    
# Helper function for plotting the distribution of buy box winners with price ranks 
# Returns priceRank
# priceRank is a dict. priceRank[i] is the number of buy box winners with price rank i
def plotPriceWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    priceRank = defaultdict(lambda: 0)
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                prices = []
                ranks = dict()
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[4])>0:
                            prices.append(float(row[4]))
                prices.sort()
                for j in range(len(prices)):
                    ranks[prices[j]] = j+1
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if len(row[4])>0:
                                    priceRank[ranks[float(row[4])]] = priceRank[ranks[float(row[4])]] + 1
            l = l+1
    return priceRank   

# Helper function for plotting the distribution of buy box winners with postive feedback ranks 
# Returns posFbRank
# posFbRank is a dict. posFbRank[i] is the number of buy box winners with postive feedback rank i
def plotPosfbWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    posFbRank = defaultdict(lambda: 0)
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                posfbs = []
                ranks = dict()
                #print(folder, i)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[7])>0:
                            posfbs.append(int(row[7].split('%')[0]))
                        j = j+1
                posfbs.sort(reverse = True)
                for j in range(len(posfbs)):
                    ranks[posfbs[j]] = j+1
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if len(row[7])>0:
                                    x = int(row[7].split('%')[0])
                                    posFbRank[ranks[x]] = posFbRank[ranks[x]] + 1
            l = l+1
    return posFbRank   

# Helper function for plotting the distribution of buy box winners with rating count ranks 
# Returns ratingCntRank
# ratingCntRank is a dict. ratingCntRank[i] is the number of buy box winners with rating count rank i  
def plotRatingCntWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    ratingCntRank = defaultdict(lambda: 0)
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                ratingCnt = []
                ranks = dict()
                #print(folder, i)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[2])>0:
                            ratingCnt.append(int(row[2]))
                        j = j+1
                ratingCnt.sort(reverse = True)
                for j in range(len(ratingCnt)):
                    ranks[ratingCnt[j]] = j+1
                #print(ranks)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if len(row[2])>0:
                                    ratingCntRank[ranks[int(row[2])]] = ratingCntRank[ranks[int(row[2])]] + 1
            l = l+1
    return ratingCntRank  

# Helper function for plotting the distribution of buy box winners with average rating ranks 
# Returns avgRatingRank
# avgRatingRank is a dict. avgRatingRank[i] is the number of buy box winners with average rating rank i  
def plotAvgRatingWinRate(mypath, amazonSold):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    avgRatingRank = defaultdict(lambda: 0)
    l = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            if amazonSold[l]:
                avgRating = []
                ranks = dict()
                #print(folder, i)
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[6])>0:
                            avgRating.append(float(row[6]))
                        j = j+1
                avgRating.sort(reverse = True)
                for j in range(len(avgRating)):
                    ranks[avgRating[j]] = j+1
                with open(folder+'/features/' + i, 'r') as featureFile:
                    read_tsv = csv.reader(featureFile, delimiter='\t')
                    j = 0
                    for row in read_tsv:
                        if j==0:
                            j = j+1
                            continue
                        if len(row[10])>0:
                            if int(row[10])==1:
                                if len(row[6])>0:
                                    avgRatingRank[ranks[float(row[6])]] = avgRatingRank[ranks[float(row[6])]] + 1
            l = l+1
    return avgRatingRank    

def printSpaces(n):
    print(' '*n, end='')
    
def main():
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams['text.usetex'] = True
    
    # Finding the Amazon sold products
    # Change the argument to the path with the backpacks-Search/backpacksOverall folder 
    amazonSold, fba = amazonSoldProducts(os.getcwd())
    print("================================ EXPERIMENT 1 ================================\n")
    
    # Plotting the distribution of products in a pie chart, i.e the number of amazon sold products, fba products, and 3p products
    no_of_products = len(amazonSold)
    data = [sum(amazonSold), sum(fba), no_of_products-sum(amazonSold)-sum(fba)]
    products = ['Amazon Sold', 'Fulfilled by Amazon', 'Third Party Sellers']
    fig = plt.figure(figsize =(6, 6))
    plt.pie(data, labels = products)
    plt.title('Products')
    # show plot
    plt.show()
    
    # Preprocessing the data (Amazon sold products only)
    # Change the argument to the path with the backpacks-Search/backpacksOverall folder 
    features, sellers = preprocess_amazon_sold_products(os.getcwd(), amazonSold)
    
    # Forming the dataset X with 7 features (Price difference to the lowest, Price ratio to the lowest, Avg rating, Positive feedback, Feedback Count, Fulfilled by Amazon?, Is Amazon the seller?
    X = features.iloc[:, [6, 7, 8, 9, 4, 10, 11]]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X)    
    # Impute our data, then train
    X = imp.transform(X)
    
    # Forming the ground truth labels (Does it win the buy box?)
    y = features.iloc[:, 12]
    
    # Plotting the distribution of sellers in a pie chart, i.e number of Amazon sellers, FBA sellers and 3P sellers 
    seller_types = ['Amazon', 'FBA', '3P Sellers']
    # Lowest price offer
    lowestPriceOffer_data = [0, 0, 0]
    l = 0
    for i in range(len(X)):
        if X[i, 0]==0.0:
            l = l+1
            if X[i, 6]==1:
                lowestPriceOffer_data[0] = lowestPriceOffer_data[0]+1
            elif X[i, 5]==1:
                lowestPriceOffer_data[1] = lowestPriceOffer_data[1]+1
            else:
                lowestPriceOffer_data[2] = lowestPriceOffer_data[2]+1
    fig = plt.figure(figsize =(6, 6))
    plt.pie(lowestPriceOffer_data, labels = seller_types)
    plt.title('Lowest offer on Price')
    # show plot
    plt.show()
    
    # Plotting the distribution of buy box winners in a pie chart, i.e number of Amazon winners, FBA winners and 3P winners 
    winBuyBox_data = [0, 0, 0]
    w = 0
    for i in range(len(X)):
        if y[i]==1:
            w = w+1
            if X[i, 6]==1:
                winBuyBox_data[0] = winBuyBox_data[0]+1
            elif X[i, 5]==1:
                winBuyBox_data[1] = winBuyBox_data[1]+1
            else:
                winBuyBox_data[2] = winBuyBox_data[2]+1
    fig = plt.figure(figsize =(6, 6))
    plt.pie(winBuyBox_data, labels = seller_types)
    plt.title('Wins Buy Box')
    # show plot
    plt.show()
    
    # Splitting into train set and test set (70-30 split)
    all_indices = [i for i in range(len(X))]
    train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=0)  # change 0.3 to 0.2 for 80-20 split
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]
    
    """sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)"""
    
    # Training the RandomForestClassifier on the train set
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    importance = clf.feature_importances_
    
    # Sorting the feature importances
    print("\nFeature importances in trained classifier\n")
    feat = ["Price difference to the lowest", "Price ratio to the lowest", "Average Rating", "Positive Feedback", "Feedback Count", "Fulfilled by Amazon?", "Is Amazon the seller?"]
    imp_sort = dict()
    for i in range(len(importance)):
        imp_sort[importance[i]] = feat[i]
    for i in sorted(imp_sort.keys(), reverse=True):
        print(imp_sort[i], end='')
        printSpaces(35-len(imp_sort[i]))
        print(i)
    print()
    
    # Testing the accuracy of classifier on test set 
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
    print("------------- Class 0 -------------")
    print("Precision:", prec[0])
    print("Recall", rec[0])
    print("F1-score:", f1[0])
    print("------------- Class 1 -------------")
    print("Precision:", prec[1])
    print("Recall", rec[1])
    print("F1-score:", f1[1])
    print("\nMacro F1:", sum(f1)/len(f1))
    
    # Grouping test samples by the number of competing sellers and computing accuracy per group
    accuracy_sellers = dict()
    lowest_price = dict()
    lowest_rank = dict()
    features_test = features.iloc[list(test_indices), :]
    grouped = features_test.groupby('Number of Sellers')
    seller_values = set(features_test['Number of Sellers'])
    for i in seller_values:
        features1 = grouped.get_group(i)
        indices = list(features1.head().index.values)
        indices1 = [test_indices.index(j) for j in indices]
        #print(indices1)
        if len(indices1)>0:
            y_test1 = np.array(y_test)[indices1]
            y_pred1 = np.array(y_pred)[indices1]
            accuracy1 = metrics.accuracy_score(y_test1, y_pred1)
            accuracy_sellers[i] = accuracy1*100
            test = features.iloc[indices]
            y_lowest_price = np.array([int(i==0) for i in test['Price difference to the lowest']])
            accuracy2 = metrics.accuracy_score(y_test1, y_lowest_price)
            lowest_price[i] = accuracy2*100
            y_lowest_rank = np.array([int(i==1) for i in test['Rank']])
            accuracy3 = metrics.accuracy_score(y_test1, y_lowest_rank)
            lowest_rank[i] = accuracy3*100
    
    # Plotting number of sellers vs accuracy for the predictions and 2 other baselines of lowest price and lowest rank    
    plt_x1 = list(accuracy_sellers.keys())
    plt_x2 = list(lowest_price.keys())
    plt_x3 = list(lowest_rank.keys())
    plt_y1 = list(accuracy_sellers.values())
    plt_y2 = list(lowest_price.values())
    plt_y3 = list(lowest_rank.values())
    fig = plt.figure(figsize=(9,6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([0, 105])
    axes.plot(plt_x1, plt_y1, linewidth='2', marker='*')
    axes.plot(plt_x2, plt_y2, 'r--', color='orange', linewidth='2', marker='s')
    axes.plot(plt_x3, plt_y3, color='green', linewidth='2', linestyle=':', marker='^')
    plt.legend (["Prediction", "Baseline: Lowest Price", "Baseline: Lowest Rank"], fontsize = 15, loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True, ncol=1)
    plt.xlabel('Number of sellers') 
    plt.ylabel('Accuracy') 
    #plt.title('Number of Sellers vs Accuracy')  
    plt.savefig('Amazon_sold_prod_no_of_sellers_vs_accuracy_70-30_split.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25)
    
    # Getting the percentage of buy box winners who satisfy certain conditions as mentioned below
    # Change the argument to the path with the backpacks-Search/backpacksOverall folder 
    priceWinRate(os.getcwd(), amazonSold)       # Lowest price
    rankWinRate(os.getcwd(), amazonSold)        # Lowest Rank
    posfbWinRate(os.getcwd(), amazonSold)       # Highest Positive feedback
    ratingCntWinRate(os.getcwd(), amazonSold)   # Highest Rating count
    avgRatingWinRate(os.getcwd(), amazonSold)   # Highest Average Rating
    
    # Plotting the plot of distribution of buy box winners with their price ranks
    priceRank = plotPriceWinRate(os.getcwd(), amazonSold)
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    plt_x3 = list(priceRank.keys())
    plt_x3.sort()
    y3 = list(priceRank.values()) 
    s = sum(y3)
    plt_y3 = []
    for i in plt_x3:
        plt_y3.append(priceRank[i]/s*100)
    axes.plot(plt_x3, plt_y3, color='red')
    plt.xlabel('Price Rank') 
    plt.ylabel('Percentage of Winners') 
    #plt.title('Distribution of winners with Price ranks')  
    plt.savefig('Amazon_sold_prod_winner_distribution_with_price_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
    # Plotting the plot of distribution of buy box winners with their ranks
    ranks = plotRankWinRate(os.getcwd(), amazonSold)
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    plt_x3 = list(ranks.keys())
    plt_x3.sort()
    y3 = list(ranks.values()) 
    s = sum(y3)
    plt_y3 = []
    for i in plt_x3:
        plt_y3.append(priceRank[i]/s*100)
    axes.plot(plt_x3, plt_y3, color='red')
    plt.xlabel('Rank') 
    plt.ylabel('Percentage of Winners') 
    #plt.title('Distribution of winners with Ranks')  
    plt.savefig('Amazon_sold_prod_winner_distribution_with_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
    # Plotting the plot of distribution of buy box winners with their positive feedback ranks
    posFbRank = plotPosfbWinRate(os.getcwd(), amazonSold)
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    plt_x4 = list(posFbRank.keys())
    plt_x4.sort()
    y4 = list(posFbRank.values()) 
    s = sum(y4)
    plt_y4 = []
    for i in plt_x4:
        plt_y4.append(posFbRank[i]/s*100)
    axes.plot(plt_x4, plt_y4, color='red')
    plt.xlabel('Positive Feedback Rank') 
    plt.ylabel('Percentage of Winners') 
    #plt.title('Distribution of winners with Positive Feedback ranks')  
    plt.savefig('Amazon_sold_prod_winner_distribution_with_posfb_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
    # Plotting the plot of distribution of buy box winners with their rating count ranks
    ratingCntRank = plotRatingCntWinRate(os.getcwd(), amazonSold)
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    plt_x5 = list(ratingCntRank.keys())
    plt_x5.sort()
    y5 = list(ratingCntRank.values()) 
    s = sum(y5)
    plt_y5 = []
    for i in plt_x5:
        plt_y5.append(ratingCntRank[i]/s*100)
    axes.plot(plt_x5, plt_y5, color='red')
    plt.xlabel('Rating Count Rank') 
    plt.ylabel('Percentage of Winners') 
    #plt.title('Distribution of winners with Rating Count ranks')  
    plt.savefig('Amazon_sold_prod_winner_distribution_with_ratingCnt_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
    # Plotting the plot of distribution of buy box winners with their average rating ranks
    avgRatingRank = plotAvgRatingWinRate(os.getcwd(), amazonSold)
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    plt_x6 = list(avgRatingRank.keys())
    plt_x6.sort()
    y6 = list(avgRatingRank.values()) 
    s = sum(y6)
    plt_y6 = []
    for i in plt_x6:
        plt_y6.append(avgRatingRank[i]/s*100)
    axes.plot(plt_x6, plt_y6, color='red')
    plt.xlabel('Average Rating Rank') 
    plt.ylabel('Percentage of Winners') 
    #plt.title('Distribution of winners with Average Rating ranks')  
    plt.savefig('Amazon_sold_prod_winner_distribution_with_avgRating_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 

if __name__=='__main__':
	main()    
