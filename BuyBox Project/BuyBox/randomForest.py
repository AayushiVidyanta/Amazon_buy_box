# -*- coding: utf-8 -*-
"""
@author: aayushi

Experiment 1
Considers the entire HEADPHONE dataset
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

# Preprocesses the feature data folder
# Returns features, sellers
# features is a pandas dataframe. Each row corresponds to a offer by a seller for a product.
# Its columns are: 'Product ID', 'Rank', 'Seller', 'Price', 'Feedback Count', 'Delivery', 'Price difference to the lowest', 'Price ratio to the lowest', 'Avg rating', 'Positive feedback', 'Fulfilled by Amazon?', 'Is Amazon the seller?', 'Does it win the buy box?', 'Number of Sellers' 
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
# 14. Number of sellers: Total number of competing sellers for that product
# =======================================================================================================================================================
# sellers is a dictionary. sellers[product ID] is the number of competing sellers for the product with that product ID
def preprocess(mypath):
    data = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #print(onlyfiles)
    sellers = dict()
    for i in onlyfiles:
        #print(i)
        if len(i.split('.'))>2:
            continue
        product_name = i.split('.')[0]
        with open(mypath + i, 'r') as featureFile:
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
                if len(row_new)==13:
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
    row_no = 0
    for i in onlyfiles:
        #print(i)
        if len(i.split('.'))>2:
            continue
        product_name = i.split('.')[0]
        with open(mypath + i, 'r') as featureFile:
            read_tsv = csv.reader(featureFile, delimiter='\t')
            list1 = list(read_tsv)
            no_of_sellers = len(list1)-1
            sellers[product_name] = no_of_sellers
            for j in range(no_of_sellers):
                data[row_no].append(no_of_sellers)
                row_no = row_no + 1
            
    #print(data)
    features = pd.DataFrame(data, columns=['Product ID', 'Rank', 'Seller', 'Price', 'Feedback Count', 'Delivery', 'Price difference to the lowest', 'Price ratio to the lowest', 'Average rating of the seller', 'Positive feedback percentage of the seller', 'Fulfilled by Amazon?', 'Is Amazon the seller?', 'Does it win the buy box?', 'Number of Sellers'])
    return features, sellers

def printSpaces(n):
    print(' '*n, end='')
    
def main():
    # Preprocessing the data
    # Change the argument to the path with the features folder 
    features, sellers = preprocess('./features/')
    
    print("================================ EXPERIMENT 1 ================================\n")
    
    # Forming the dataset X with 7 features (Price difference to the lowest, Price ratio to the lowest, Avg rating, Positive feedback, Feedback Count, Fulfilled by Amazon?, Is Amazon the seller?
    #features.drop_duplicates(keep='first', inplace=True, ignore_index=True) 
    X = features.iloc[:, [6, 7, 8, 9, 4, 10, 11]]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X)
    # Impute our data, then train
    X = imp.transform(X)
    
    # Forming the ground truth labels (Does it win the buy box?)
    y = features.iloc[:, 12]
    
    # Splitting into train set and test set (70-30 split)
    all_indices = [i for i in range(len(X))]
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=0)  # change 0.3 to 0.2 for 80-20 split
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
    print("Feature importances in trained classifier\n")
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
    axes.plot(plt_x1, plt_y1, linewidth='2')
    axes.plot(plt_x2, plt_y2, 'r--', color='orange', linewidth='2')
    axes.plot(plt_x3, plt_y3, color='green', linewidth='2', linestyle=':')
    plt.legend(["Prediction", "Baseline: Lowest Price", "Baseline: Lowest Rank"])
    plt.xlabel('Number of sellers') 
    plt.ylabel('Accuracy') 
    plt.title('Number of Sellers vs Accuracy')  
    plt.show() 

if __name__=='__main__':
	main()   
