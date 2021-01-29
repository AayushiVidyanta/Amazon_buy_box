#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:20:26 2020

@author: aayushi

Experiment 1
Considers the entire BACKPACK dataset
"""

import csv
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt 

# Preprocesses the feature data folder
# Returns rank_winners, rank_sellers, rank_win_rate, posfb_sellers, avgRating_sellers
# rank_winners is a dict. rank_winners[i] is the number of winners of the buy box with rank i
# rank_sellers is a dict. rank_sellers[i] is the number of competing sellers with rank i
# rank_win_rate is a dict. rank_win_rate[i] is the win rate of sellers with rank i. rank_win_rate[i] = rank_winners[i]/rank_sellers[i]
# posfb_sellers is a dict. posfb_sellers[i] is the number of sellers with positive feedback = i
# avgRating_sellers is a dict. avgRating_sellers[i] is the number of sellers with average rating <= i
def preprocess(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    rank_winners = defaultdict(lambda: 0)
    rank_sellers = defaultdict(lambda: 0)
    posfb_sellers = defaultdict(lambda: 0)
    avgRating_sellers = dict()
    avgRating_sellers[0] = 0
    avgRating_sellers[0.5] = 0
    avgRating_sellers[1] = 0
    avgRating_sellers[1.5] = 0
    avgRating_sellers[2] = 0
    avgRating_sellers[2.5] = 0
    avgRating_sellers[3] = 0
    avgRating_sellers[3.5] = 0
    avgRating_sellers[4] = 0
    avgRating_sellers[4.5] = 0
    avgRating_sellers[5] = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
            with open(folder+'/features/' + i, 'r') as featureFile:
                read_tsv = csv.reader(featureFile, delimiter='\t')
                j = 0
                for row in read_tsv:
                    if j==0:
                        j = j+1
                        continue
                    rank_sellers[j] = rank_sellers[j]+1
                    if len(row[10])>0:
                        #print(row[10])
                        if int(row[10])==1:
                            rank_winners[j] = rank_winners[j]+1                                    
                    if len(row[7])>0:
                        posfb = int(row[7].split('%')[0])
                        posfb_sellers[posfb] = posfb_sellers[posfb]+1
                    if len(row[6])>0:
                        avgRating = float(row[6])
                        if avgRating <= 0:
                            avgRating_sellers[0] = avgRating_sellers[0] + 1
                        if avgRating <= 0.5:
                            avgRating_sellers[0.5] = avgRating_sellers[0.5] + 1
                        if avgRating <= 1:
                            avgRating_sellers[1] = avgRating_sellers[1] + 1
                        if avgRating <= 1.5:
                            avgRating_sellers[1.5] = avgRating_sellers[1.5] + 1
                        if avgRating <= 2:
                            avgRating_sellers[2] = avgRating_sellers[2] + 1
                        if avgRating <= 2.5:
                            avgRating_sellers[2.5] = avgRating_sellers[2.5] + 1
                        if avgRating <= 3:
                            avgRating_sellers[3] = avgRating_sellers[3] + 1
                        if avgRating <= 3.5:
                            avgRating_sellers[3.5] = avgRating_sellers[3.5] + 1
                        if avgRating <= 4:
                            avgRating_sellers[4] = avgRating_sellers[4] + 1 
                        if avgRating <= 4.5:
                            avgRating_sellers[4.5] = avgRating_sellers[4.5] + 1
                        if avgRating <= 5:
                            avgRating_sellers[5] = avgRating_sellers[5] + 1
                        
                    j = j+1
        rank_win_rate = defaultdict(lambda: 0)
        for i in rank_sellers.keys():
            rank_win_rate[i] = rank_winners[i]/rank_sellers[i]*100
    
    return rank_winners, rank_sellers, rank_win_rate, posfb_sellers, avgRating_sellers          

# Prints the percentage of buy box winners which offer the lowest price
def priceWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
    t = t0+t1
    print("Lowest Price")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))  
    
# Helper function for plotting the distribution of buy box winners with price ranks 
# Returns priceRank
# priceRank is a dict. priceRank[i] is the number of buy box winners with price rank i
def plotPriceWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    priceRank = defaultdict(lambda: 0)
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
    return priceRank                                    

# Prints the percentage of buy box winners which have the lowest rank
def rankWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
    t = t0+t1
    print("Lowest Rank")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))                            
    
# Prints the percentage of buy box winners which have the highest positive feedback
def posfbWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
            if len(posfbs):
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
    t = t0+t1
    print("Highest Postive Feedback")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))
    
# Helper function for plotting the distribution of buy box winners with postive feedback ranks 
# Returns posFbRank
# posFbRank is a dict. posFbRank[i] is the number of buy box winners with postive feedback rank i
def plotPosfbWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    posFbRank = defaultdict(lambda: 0)
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
    return posFbRank                                    
    
# Prints the percentage of buy box winners which have the highest rating count
def ratingCntWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
            if len(ratingCnt):
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
    t = t0+t1
    print("Highest Rating Count")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))

# Helper function for plotting the distribution of buy box winners with rating count ranks 
# Returns ratingCntRank
# ratingCntRank is a dict. ratingCntRank[i] is the number of buy box winners with rating count rank i  
def plotRatingCntWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    ratingCntRank = defaultdict(lambda: 0)
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
    return ratingCntRank                 

# Prints the percentage of buy box winners which have the highest average rating
def avgRatingWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    t0 = 0
    t1 = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
            if len(avgRat):
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
    t = t0+t1
    print("Highest Average Rating")
    print("         1")
    print(" 0   |", round(t0/t*100, 2))
    print(" 1   |", round(t1/t*100, 2))  

# Helper function for plotting the distribution of buy box winners with average rating ranks 
# Returns avgRatingRank
# avgRatingRank is a dict. avgRatingRank[i] is the number of buy box winners with average rating rank i  
def plotAvgRatingWinRate(mypath):
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    avgRatingRank = defaultdict(lambda: 0)
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
    	#print(onlyfiles)
        for i in onlyfiles:
    		#print(i)
            if len(i.split('.'))>2:
                continue
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
    return avgRatingRank                   
                            
def main():
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams['text.usetex'] = True
    
    # Preprocessing the data
    # Change the argument to the path with the backpacks-Search/backpacksOverall folder 
    rank_winners, rank_sellers, rank_win_rate, posfb_sellers, avgRating_sellers = preprocess(os.getcwd())
    
    print("================================ EXPERIMENT 1 ================================\n")
    
    # Plotting the Rank vs Win-Rate % plot 
    fig = plt.figure(figsize=(9,6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    x = list(rank_win_rate.keys())
    y = list(rank_win_rate.values())
    axes.plot(x, y, color='red')
    plt.xlabel('Rank') 
    plt.ylabel('Win-Rate %') 
    #plt.title('Rank vs Win-Rate')  
    plt.savefig('All_prod_rank_vs_win_rate.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25)
    
    # Plotting the cumulative plot of Positive Feedback vs Win-Rate
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    plt_x1 = list(posfb_sellers.keys())
    plt_x1.sort()
    y1 = []
    for i in plt_x1:
        y1.append(posfb_sellers[i])
    plt_y1 = [] 
    s = sum(y1)
    for i in range(len(y1)):
        plt_y1.append(sum(y1[:i+1])/s*100)
    axes.plot(plt_x1, plt_y1, color='red')
    plt.xlabel('Positive Feedback') 
    plt.ylabel('Win-Rate %') 
    plt.title('Cumulative Plot : Positive Feedback vs Win-Rate')  
    plt.show() 
    
    # Plotting the cumulative plot of Average Rating vs Win-Rate
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    plt_x2 = list(avgRating_sellers.keys())
    y2 = list(avgRating_sellers.values()) 
    s = y2[-1]
    plt_y2 = []
    for i in y2:
        plt_y2.append(i/s*100)
    axes.plot(plt_x2, plt_y2, color='red')
    plt.xlabel('Average Rating') 
    plt.ylabel('Win-Rate %') 
    plt.title('Cumulative Plot : Average Rating vs Win-Rate')  
    plt.show() 
    
    # Getting the percentage of buy box winners who satisfy certain conditions as mentioned below
    # Change the argument to the path with the backpacks-Search/backpacksOverall folder 
    priceWinRate(os.getcwd())       # Lowest price
    rankWinRate(os.getcwd())        # Lowest Rank
    posfbWinRate(os.getcwd())       # Highest Positive feedback
    ratingCntWinRate(os.getcwd())   # Highest Rating count
    avgRatingWinRate(os.getcwd())   # Highest Average Rating
    
    # Plotting the plot of distribution of buy box winners with their price ranks
    priceRank = plotPriceWinRate(os.getcwd())
    fig = plt.figure(figsize=(9,6))
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
    plt.savefig('All_prod_winner_distribution_with_price_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
    # Plotting the plot of distribution of buy box winners with their positive feedback ranks
    posFbRank = plotPosfbWinRate(os.getcwd())
    fig = plt.figure(figsize=(9,6))
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
    plt.savefig('All_prod_winner_distribution_with_posfb_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
    # Plotting the plot of distribution of buy box winners with their rating count ranks
    ratingCntRank = plotRatingCntWinRate(os.getcwd())
    fig = plt.figure(figsize=(9,6))
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
    plt.savefig('All_prod_winner_distribution_with_ratingCnt_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
    # Plotting the plot of distribution of buy box winners with their average rating ranks
    avgRatingRank = plotAvgRatingWinRate(os.getcwd())
    fig = plt.figure(figsize=(9,6))
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
    plt.savefig('All_prod_winner_distribution_with_avgRating_rank.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25) 
    
if __name__=='__main__':
    main()    