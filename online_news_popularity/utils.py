def clean_cols(data):
    """Clean the column names by stripping and lowercase."""
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(index=str, columns=clean_col_map)

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
    return np.average(length)

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
    return global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,min_positive_polarity,
            max_positive_polarity,avg_negative_polarity,min_negative_polarity,max_negative_polarity









