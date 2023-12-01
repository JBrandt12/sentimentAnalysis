import pandas as pd 
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer 
import os
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

sia = SentimentIntensityAnalyzer() 
def analyzier(newshealines): 
    res = []
    for i in newshealines: 
        text = sia.polarity_scores(i) 
        res.append(text) 
    test = pd.DataFrame(res) 
    greaterThanpos = test['pos'] > 0
    x = test['pos'][greaterThanpos].mean()

    greatthanneg = test['neg'] > 0 
    y = test['neg'][greatthanneg].mean()
    if pd.isna(x): 
        x = 0
    if pd.isna(y): 
        y=0 
    return (x,y)


