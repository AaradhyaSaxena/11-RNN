import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="white")
from time import time
from sklearn.metrics import jaccard_score, accuracy_score, fbeta_score, roc_curve, auc, roc_auc_score

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def clean_cols(data):
    """Clean the column names by stripping and lowercase."""
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(columns=clean_col_map)

def TrainTestSplit(X, Y, R=0, test_size=0.2):
    """Easy Train Test Split call."""
    return train_test_split(X, Y, test_size=test_size, random_state=R)

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))

def rate_unique(words):
    words=tokenize(words)
    no_order = list(set(words))
    rate_unique=len(no_order)/len(words)
    return rate_unique

def rate_nonstop(words):
    words=tokenize(words)
    filtered_sentence = [w for w in words if not w in stopwords]
    rate_nonstop=len(filtered_sentence)/len(words)
    no_order = list(set(filtered_sentence))
    rate_unique_nonstop=len(no_order)/len(words)
    return rate_nonstop,rate_unique_nonstop

def avg_token(words):
    words=tokenize(words)
    length=[]
    for i in words:
        length.append(len(i))
    ans = sum(length)/len(length)
    return ans

from textblob import TextBlob
import datefinder
import datetime  
from datetime import date

def day(article_text):
    article=article_text
    if len(list(datefinder.find_dates(article)))>0:
        date=str(list(datefinder.find_dates(article))[0])
        date=date.split()
        date=date[0]
        year, month, day = date.split('-')     
        day_name = datetime.date(int(year), int(month), int(day)) 
        return day_name.strftime("%A")
    return "Monday"

def tokenize(text):
    text=text
    return word_tokenize(text)

pos_words=[]
neg_words=[]

def polar(words):
    all_tokens=tokenize(words)
    for i in all_tokens:
        analysis=TextBlob(i)
        polarity=analysis.sentiment.polarity
        if polarity>0:
            pos_words.append(i)
        if polarity<0:
            neg_words.append(i)
    return pos_words,neg_words

def rates(words):
    words=polar(words)
    pos=words[0]
    neg=words[1]
    all_words=words
    global_rate_positive_words=(len(pos)/len(all_words))/100
    global_rate_negative_words=(len(neg)/len(all_words))/100
    pol_pos=[]
    pol_neg=[]
    for i in pos:
        analysis=TextBlob(i)
        pol_pos.append(analysis.sentiment.polarity)
        avg_positive_polarity=analysis.sentiment.polarity
    for j in neg:
        analysis2=TextBlob(j)
        pol_neg.append(analysis2.sentiment.polarity)
        avg_negative_polarity=analysis2.sentiment.polarity
    min_positive_polarity=min(pol_pos)
    max_positive_polarity=max(pol_pos)
    min_negative_polarity=min(pol_neg)
    max_negative_polarity=max(pol_neg)
    avg_positive_polarity=np.average(pol_pos)
    avg_negative_polarity=np.average(pol_neg)
    return global_rate_positive_words, global_rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity, avg_negative_polarity, min_negative_polarity, max_negative_polarity


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    results = {}
    
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time

    results['train_time'] = end-start
        
    # Get predictions on the first 4000 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:4000])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end-start
    # Compute accuracy on the first 4000 training samples
    results['acc_train'] = jaccard_score(y_train[:4000],predictions_train, average='weighted')
    # Compute accuracy on test set
    results['acc_test'] = jaccard_score(y_test,predictions_test, average='weighted')
    # Compute F-score on the the first 4000 training samples
#     results['f_train'] = fbeta_score(y_train[:4000],predictions_train,beta=1)
#     # Compute F-score on the test set
#     results['f_test'] = fbeta_score(y_test,predictions_test,beta=1)
#     # Compute AUC on the the first 4000 training samples
#     results['auc_train'] = roc_auc_score(y_train[:4000],predictions_train)
#     # Compute AUC on the test set
#     results['auc_test'] = roc_auc_score(y_test,predictions_test)
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
#     print( "{} with accuracy {}, F1 {} and AUC {}.".format(learner.__class__.__name__,results['acc_test'],results['f_test'], results['auc_test']))   
    
    return results











