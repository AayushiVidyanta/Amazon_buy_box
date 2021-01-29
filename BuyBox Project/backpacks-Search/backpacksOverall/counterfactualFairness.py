#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:09:24 2021

@author: aayushi

Experiment 2 and 3
Considers the entire BACKPACK dataset
"""

import pandas as pd
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import os 
from collections import defaultdict

# Preprocesses the feature data folder
# Returns features, sellers, winAmazon
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
# winAmazon is the number of times Cloudtail India or Appario Retail Private Ltd wins the buy box in the entire dataset
def preprocess(mypath):
    data = []
    row_no = 0
    folders = [f for f in listdir(mypath) if os.path.isdir(mypath+'/'+f)]
    winAmazon = 0
    for folder in folders:
        onlyfiles = [f for f in listdir(folder+'/features/') if isfile(join(folder+'/features/', f))]
        #print(onlyfiles)
        sellers = dict()
        for i in onlyfiles:
            #print(i)
            if len(i.split('.'))>2:
                continue
            product_name = i.split('.')[0]
            with open(folder+'/features/' + i, 'r') as featureFile:
                #print(folder + " " + i)
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
                        if row_new[11]==1 and row_new[12]==1:
                            winAmazon = winAmazon+1
                        #print(row_new)
                        data.append(row_new)
                        j = j+1
        
        for i in onlyfiles:
            #print(i)
            if len(i.split('.'))>2:
                continue
            product_name = i.split('.')[0]
            with open(folder+'/features/' + i, 'r') as featureFile:
                read_tsv = csv.reader(featureFile, delimiter='\t')
                list1 = list(read_tsv)
                no_of_sellers = len(list1)-1
                sellers[product_name] = no_of_sellers
                for j in range(no_of_sellers):
                    data[row_no].append(no_of_sellers)
                    row_no = row_no + 1
                
    features = pd.DataFrame(data, columns=['Product ID', 'Rank', 'Seller', 'Price', 'Feedback Count', 'Delivery', 'Price difference to the lowest', 'Price ratio to the lowest', 'Avg rating', 'Positive feedback', 'Fulfilled by Amazon?', 'Is Amazon the seller?', 'Does it win the buy box?', 'Timestamp', 'Number of Sellers'])
    return features, sellers, winAmazon

def printSpaces(n):
    print(' '*n, end='')
    
def main():
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams['text.usetex'] = True
    
    # Preprocessing the data
    # Change the argument to the path with the backpacks-Search/backpacksOverall folder
    features, sellers, winAmazon = preprocess(os.getcwd()) 
    
    # Forming the dataset X with 7 features (Price difference to the lowest, Price ratio to the lowest, Avg rating, Positive feedback, Feedback Count, Fulfilled by Amazon?, Is Amazon the seller?
    X = features.iloc[:, [6, 7, 8, 9, 4, 10, 11]]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X)
    # Impute our data, then train
    X = imp.transform(X)
    
    # Forming the ground truth labels (Does it win the buy box?)
    y = features.iloc[:, 12]
    
    # Splitting into train set and test set (70-30 split)
    all_indices = [i for i in range(len(X))]
    train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=0)  # change 0.3 to 0.2 for 80-20 split
    X_train = X[train_indices, :]
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test_org = X[test_indices, :]
    y_test = y[test_indices]
    
    """sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)"""
    
    # Training the RandomForestClassifier on the train set
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    importance = clf.feature_importances_
    
    # Sorting the feature importances
    print("================================ EXPERIMENT 2 ================================\n")
    feat = ["Price difference to the lowest", "Price ratio to the lowest", "Average Rating", "Positive Feedback", "Feedback Count", "Fulfilled by Amazon?", "Is Amazon the seller?"]
    imp_sort = dict()
    print("Feature importances in trained classifier\n")
    for i in range(len(importance)):
        imp_sort[importance[i]] = feat[i]
    for i in sorted(imp_sort.keys(), reverse=True):
        print(imp_sort[i], end='')
        printSpaces(35-len(imp_sort[i]))
        print(i)
    print()
    
    # Testing the accuracy of classifier on original test set 
    y_pred_org = clf.predict(X_test_org)
    accuracy_org = metrics.accuracy_score(y_test, y_pred_org)
    print("Accuracy on original test set:", accuracy_org)  
    
    # Finding the TPR (True Positive Rate), FPR (False Positive Rate) for various thresholds on original test set
    y_probs_org = clf.predict_proba(X_test_org)
    fpr_org, tpr_org, threshold_org = metrics.roc_curve(y_test, y_probs_org[:, 1])
    
    # Counterfactual pairs replacement (Expreiment 2)
    # Forming the counterfactual test set by setting the value of feature 'Is Amazon the seller?' of all the test samples with 'Is Amazon the seller?' as 1 to 0, keeping the ground truth label unchanged
    X_test_ctf = X_test_org.copy()
    for i in range(len(X_test_ctf)):
        if X_test_ctf[i][6]:
            X_test_ctf[i][6] = 0
    
    # Testing the accuracy of classifier on counterfactual test set 
    y_pred_ctf = clf.predict(X_test_ctf)
    accuracy_ctf = metrics.accuracy_score(y_test, y_pred_ctf)
    print("Accuracy on counterfactual test set:", accuracy_ctf)
    
    # Finding the TPR (True Positive Rate), FPR (False Positive Rate) for various thresholds on counterfactual test set
    y_probs_ctf = clf.predict_proba(X_test_ctf)
    fpr_ctf, tpr_ctf, threshold_ctf = metrics.roc_curve(y_test, y_probs_ctf[:, 1])
    
    # Reporting the CTF Gap
    amazonOffers = 0
    amazonWon = 0
    amazonPred_org = 0
    amazonPred_ctf = 0
    avg = 0
    for i in range(len(X_test_org)):
        if X_test_org[i][6]:
           avg = avg+abs(y_pred_ctf[i]-y_pred_org[i]) 
           amazonOffers = amazonOffers+1
           if y_test[test_indices[i]]:
               amazonWon = amazonWon+1
           if y_pred_ctf[i]:
               amazonPred_org = amazonPred_org+1
           if y_pred_ctf[i]:
               amazonPred_ctf = amazonPred_ctf+1
    ctf_gap = avg/amazonOffers
    print("CTF Gap is", ctf_gap)
    print("Number of offers in test set is", len(test_indices))
    print("Number of offers by Amazon in test set is", amazonOffers)
    print("Number of offers by Amazon winning the buybox in the test set is", amazonWon)
    print("Number of offers by Amazon predicted to win the buybox in the original test set is", amazonPred_org)
    print("Number of offers by Amazon predicted to win the buybox in the counterfactual test set is", amazonPred_ctf)
    
    # Grouping test samples by the number of competing sellers and computing accuracy per group
    accuracy_sellers = dict()
    lowest_price = dict()
    lowest_rank = dict()
    features_test = features.iloc[list(test_indices), :]
    grouped = features_test.groupby('Number of Sellers')
    seller_values = set(features_test['Number of Sellers'])
    for i in seller_values:
        #print(i)
        features1 = grouped.get_group(i)
        indices = list(features1.head().index.values)
        indices1 = [test_indices.index(j) for j in indices]
        #print(indices1)
        if len(indices1)>0:
            y_test1 = np.array(y_test)[indices1]
            y_pred1 = np.array(y_pred_ctf)[indices1]
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
    plt.title('Number of Sellers vs Accuracy')  
    plt.savefig('All_prod_no_of_sellers_vs_accuracy_70-30_split_CF.pdf', transparent= True, bbox_inches='tight', dpi = 500, pad_inches = 0.25)
    
    # Chi-square test (Experiment 3)
    # Regularizing the train set with MinMaxScaler
    X_train_reg = X_train.copy()
    mm = MinMaxScaler()
    X_train_reg = mm.fit_transform(X_train_reg)
    
    # Finding the chi-square and statistical importances of the 7 features
    F, pval = chi2(X_train_reg, y_train)
    
    # Sorting the feature importances
    print("\n================================ EXPERIMENT 3 ================================\n")
    chi_sort = dict()
    print("Feature importances in trained classifier\n")
    for i in range(len(F)):
        chi_sort[F[i]] = [feat[i], pval[i]]
    for i in sorted(chi_sort.keys(), reverse=True):
        print(chi_sort[i][0], end='')
        printSpaces(35-len(chi_sort[i][0]))
        print(i, "\t", chi_sort[i][1])
    print()
    
    # Training the RandomForestClassifier on the regularized train set
    clf1 = RandomForestClassifier(n_estimators=100, random_state=0)
    clf1.fit(X_train_reg, y_train)
    
    # Testing the accuracy of classifier on regularized (by MinMaxScalar) test set 
    X_test_reg = X_test_org.copy()
    mm = MinMaxScaler()
    X_test_reg = mm.fit_transform(X_test_reg)
    y_pred_reg = clf.predict(X_test_reg)
    accuracy_reg = metrics.accuracy_score(y_test, y_pred_reg)
    print("Accuracy on regularized test set:", accuracy_reg)
    
if __name__=='__main__':
	main()
