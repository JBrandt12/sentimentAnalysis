import pandas as pd 
import numpy as np 
import nltk 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('ggplot') 
from nltk.sentiment import SentimentIntensityAnalyzer 

sia = SentimentIntensityAnalyzer() 
# newshealines = ["Bitcoin Tops $37K for First Time Since May 2022 as Short Squeeze Bumps Prices Amid BTC ETF Optimism","UBS Group’s Wealthy Clients Can Now Trade Some Crypto ETFs in Hong Kong: Bloomberg","2 Years Ago, Bitcoin Hit an All-Time High. Is Another Rally on the Way?", "FTX's FTT Token Jumps 90% on Gensler Comments" , "U.S. SEC Said to Open Talks with Grayscale on Spot Bitcoin ETF Push"]

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


# analyzier(newshealines) 


