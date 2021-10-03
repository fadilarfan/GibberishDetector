# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 23:54:33 2021

@author: Heliotrope
"""
import pickle
import numpy as np
from tqdm import tqdm
import math

## Input the text training data, the good words list, and the bad words list
trainCorpusAll = 'allData.txt'
good = 'good.txt'
bad = 'bad.txt'
acceptedChars = 'abcdefghijklmnopqrstuvwxyz '
lenAcceptedChars = len(acceptedChars)
charIndex = {c: idx for idx, c in enumerate(acceptedChars)}

def Tokenize(text):
    return [char.lower() for char in text if char.lower() in acceptedChars]

def NGram(n, txt):
    tokenized_txt = Tokenize(txt)
    for idx in range(len(tokenized_txt)-n+1):
        yield tokenized_txt[idx:idx+n]
        
def predict(text, probabilities, charIndex):
    pb = 0
    count = 0
    for firstChar, secondChar in NGram(2, text):
        pb += probabilities[charIndex[firstChar]][charIndex[secondChar]]
        count+=1
    return math.exp(pb/(count or 1))

def Train(trainCorpus, filename):
    probabilities = np.ones((27,27))*1
    for line in tqdm(open(trainCorpus, 'r',encoding="utf8")):
        for firstChar, secChar in NGram(2, line):
            probabilities[charIndex[firstChar]][charIndex[secChar]] += 1
    for i, row in enumerate(probabilities):
        s = float(sum(row))
        for j in range(len(row)):
            print(row[j])
            row[j] = math.log(row[j]/s)
            print(row[j])
    filename1 = filename
    pickle.dump(probabilities, open(filename1, 'wb'))
    findThreshold(charIndex, good, bad, 'thresholdAll.sav', 'modelAll.sav')
   
    
def findThreshold(charIndex, filenameGood, filenameBad, filenameThresh, trainedModel):
    threshLow = float('inf')
    threshHigh = float('-inf')
    loadedModel = pickle.load(open(trainedModel, 'rb'))
    for line in tqdm(open(filenameGood, 'r',encoding="utf8")):
        for firstChar, secChar in NGram(2, line):
            predictValue = predict(line, loadedModel, charIndex)
            if predictValue < threshLow:
                threshLow = predictValue
    for line in tqdm(open(filenameBad, 'r',encoding="utf8")):
        for firstChar, secChar in NGram(2, line):
            predictValue = predict(line, loadedModel, charIndex)
            if predictValue > threshHigh:
                threshHigh = predictValue                   
    thresholdFinal = (threshLow + threshHigh)/2
    filename2 = filenameThresh
    pickle.dump(thresholdFinal, open(filename2, 'wb'))

## Just run this when you need to train the model and comment it anytime you don't need it
#Train(trainCorpusAll, 'modelAll.sav')

    

