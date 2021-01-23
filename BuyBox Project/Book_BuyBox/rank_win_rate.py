#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:17:55 2020

@author: aayushi

Experiment 1
Considers the entire BOOK dataset
"""

import csv
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import matplotlib.pyplot as plt 

# Preprocesses the feature data folder
# Returns rank_winners, rank_sellers, rank_win_rate
# rank_winners is a dict. rank_winners[i] is the number of winners of the buy box with rank i
# rank_sellers is a dict. rank_sellers[i] is the number of competing sellers with rank i
# rank_win_rate is a dict. rank_win_rate[i] is the win rate of sellers with rank i. rank_win_rate[i] = rank_winners[i]/rank_sellers[i]
def preprocess(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	#print(onlyfiles)
    rank_winners = defaultdict(lambda: 0)
    rank_sellers = defaultdict(lambda: 0)
    for i in onlyfiles:
		#print(i)
        if len(i.split('.'))>2:
            continue
        with open(mypath + i, 'r') as featureFile:
            read_tsv = csv.reader(featureFile, delimiter='\t')
            j = 0
            for row in read_tsv:
                if j==0:
                    j = j+1
                    continue
                rank_sellers[j] = rank_sellers[j]+1
                if len(row[9])>0:
                    #print(row[10])
                    if int(row[9])==1:
                        rank_winners[j] = rank_winners[j]+1
                j = j+1
    rank_win_rate = defaultdict(lambda: 0)
    for i in rank_sellers.keys():
        rank_win_rate[i] = rank_winners[i]/rank_sellers[i]*100
    
    return rank_winners, rank_sellers, rank_win_rate                  

def main():
    # Preprocessing the data
    # Change the argument to the path with the buybox folder
    rank_winners, rank_sellers, rank_win_rate = preprocess('./features/')
    
    print("================================ EXPERIMENT 1 ================================\n")
    
    # Plotting the Rank vs Win-Rate % plot 
    fig = plt.figure(figsize=(7.5,5))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylim([-2, 105])
    x = list(rank_win_rate.keys())
    y = list(rank_win_rate.values())
    axes.plot(x, y, color='red')
    plt.xlabel('Rank') 
    plt.ylabel('Win-Rate %') 
    plt.title('Rank vs Win-Rate')  
    plt.show() 
    
if __name__=='__main__':
	main()    
