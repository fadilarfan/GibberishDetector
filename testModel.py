# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 00:37:38 2021

@author: Heliotrope
"""
import csv
import pickle
import model
import time
## Input the slang words text data and load both the probability model and threshold value from the model
acceptedChars = 'abcdefghijklmnopqrstuvwxyz '
charIndex = {c: idx for idx, c in enumerate(acceptedChars)}
slangFile = open('slangList.txt','r')
slangList = slangFile. read()
loadedModel = pickle.load(open('modelAll.sav', 'rb'))
loadedThreshold = pickle.load(open('thresholdAll.sav', 'rb'))

## Input the test data from a csv file
with open('Gibberish Word Data - Gibberish Data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    wordDict = dict()
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            
        else:
            line_count +=1
            wordDict[row[1]] = row[2]
            
def PredictAll(textDict, probabilities, charIndex, threshold):
    resultsAll = dict()
    for i in textDict:
        prediction = model.predict(i, probabilities, charIndex)
        if prediction > loadedThreshold:
            if i not in slangList:
                resultsAll[i] = 'Not Gibberish'
            else:
                resultsAll[i] = 'Gibberish'
        else:
            resultsAll[i] = 'Gibberish'
    return resultsAll

## Optional method only used to monitor the performance of the system
def validasi(testDict, trueDict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in testDict:
        if testDict[i] == trueDict[i]:
            if testDict[i] == 'Gibberish':
                TN+=1
            else:
                TP+=1
        else:
            if testDict[i] == 'Gibberish':
                FN+=1
            else:
                FP+=1      
    confusionMatrix =[
                     [TP, FN],
                     [FP, TN]
                     ]
    print("    NG  G")
    print("NG " + str(confusionMatrix[0]))
    print("G  " + str(confusionMatrix[1]))
    print('akurasi = '+ str((TP+TN)/(TP+FP+FN+TN)))

## To run the model through the test data and also time the runtime
start = time.time()
prediction = PredictAll(wordDict, loadedModel, model.charIndex, loadedThreshold)
stop = time.time()
print("\nRuntime: " + str(stop-start))
print('-=Prediction done=-')





