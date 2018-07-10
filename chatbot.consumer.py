#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:52:05 2018

@author: ambikaajaggi
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid
from termcolor import colored

senti_analyzer = sid()

# =============================================================================
# Initiate conversation and determine sentiment from consumer initial response
# =============================================================================

neg = False
print(colored('Hi my name is Alex! I noticed you recently purchased a product,' 
              ' how was it?', 'blue'))
userInput1 = str(input(">>> "))
print('')
if senti_analyzer.polarity_scores(userInput1)['compound']<0:
        neg = True

sentiment = senti_analyzer.polarity_scores(userInput1)['compound']

import pickle

# =============================================================================
# Use negative or positive trained classifiers based on consumer sentiment such
# that the chatbot can reciprocate the sentiment whilst also predicting the
# star rating
# =============================================================================

if neg == False: 
    classifier2_f = open("LinearSVC.pickle","rb")
    classifier2 = pickle.load(classifier2_f)
    classifier2_f.close()
    preds = classifier2.predict(sentiment)
    print(colored("That's great! So would you give it" , 'blue'),
          colored(preds[0], 'blue'), colored('stars?', 'blue'))
else:
    classifier2_f = open("NEGLinearSVC.pickle","rb")
    classifier2 = pickle.load(classifier2_f)
    classifier2_f.close()
    preds = classifier2.predict(sentiment)    
    print(colored("Oh no! So would you give it", 'blue'), 
          colored(preds[0], 'blue'), colored('stars?', 'blue'))

# =============================================================================
# Seeries of steps to determine whether the prediction was accurate or not, 
# providing a counter prediction and if both of those fail, reading and 
# understanding th user inputter star rating
# =============================================================================

userInput2 = input(">>> ")
print('')
if senti_analyzer.polarity_scores(userInput2)['compound']>0:
    rating = preds[0]
else:
    if preds != 1:
        new_preds = preds[0]-1
    else:
        new_preds = preds[0] + 1
    print(colored('No? Ok how about', 'blue'), colored(new_preds, 'blue'), 
          colored('stars?', 'blue'))
    userInput3 = input(">>> ")
    print('')
    if senti_analyzer.polarity_scores(userInput3)['compound']>0:
        rating = new_preds
        next_step = True
    else:
        print(colored("No? Well maybe it's easier if you tell me", 'blue'))
        proper_value = False
        while proper_value == False:
            rating = input('>>> ')
            print('')
            if '.' in rating:
                rating = round(float(rating))
                proper_value = True
            elif any(i in rating for i in '12345') == True:
                rating = int(rating)
                proper_value = True
            else:
                try:
                    from word2number import w2n
                    rating = w2n.word_to_num(rating) 
                    proper_value = True
                except (TypeError, ValueError):
                    print(colored("I'm sorry I couldn't understand, please "
                                  "give me an integer between and including 1 "
                                  "and 5", 'blue'))
    if rating>5:
        rating = 5
    elif rating<1:
        rating = 1  

# =============================================================================
# Propmting for a more detailed product review
# =============================================================================

check_for_product = False
print(colored('Great! Would you like to give a more detailed review of the '
              'product?', 'blue'))
userInput4 = input(">>> ")
print('')
if senti_analyzer.polarity_scores(userInput4)['compound']>0:
    print(colored('Great! Type away:', 'blue'))
    review = input(">>> ")
    print('')
    check_for_product = True
elif senti_analyzer.polarity_scores(userInput4)['compound']<0:
    print(colored('Are you sure, your review can help improve this product in '
                  'the future?', 'blue'))
    userInput5 = input(">>> ")
    print('')
    if senti_analyzer.polarity_scores(userInput5)['compound']>0.25:
        print(colored('Okay, thank you for your time!', 'blue'))
    else:
        print(colored('Great! Type away:', 'blue'))
        review = input(">>> ")
        print('')
        check_for_product = True

# =============================================================================
# If willing to give a review, predict the product from that given review, 
# prompting a user for more information to be used as key words if necesary
# =============================================================================

if check_for_product == True:   
    classifier3_f = open("MultinomialNB.pickle","rb")
    classifier3 = pickle.load(classifier3_f)
    classifier3_f.close()
    
    vectorizer_f = open("CountVectorizer.pickle","rb")
    vectorizer = pickle.load(vectorizer_f)
    vectorizer_f.close()
    
    product = classifier3.predict(vectorizer.transform([review]))
    
    print(colored('Just to confirm, you purchased', 'blue'),
          colored(''.join(product), 'blue'),colored('right?', 'blue'))
    userInput6 = input(">>> ")
    print('')
    if senti_analyzer.polarity_scores(userInput6)['compound']>0:
        print(colored('Great you are all set, thank you and have a great day!', 
                      'blue'))
    else:
        right_product = False
        old_prod = []
        import os
        import pandas as pd
        import nltk
        cwd = os.getcwd()  # Gets the current working directory (cwd)
        files = os.listdir(cwd)  # Gets all the files in that directory
        df=pd.read_csv('GrammarandProductReviews.csv')
        import difflib
        print(colored("I'm sorry I couldn't understand, could you give me the "
                      "exact name?", 'blue'))
        old_prod.append(product)
        product = (input(">>> ")).lower()
        prod_list = []
        for j in range(len(df['name'])):
            if nltk.tokenize.word_tokenize(product)[0] in df['name'][j].lower():
                prod_list.append(df['name'][j].lower())
        prod_list = list(set(prod_list))
        while right_product == False:
            for i in old_prod:
                if i in prod_list:
                    prod_list.remove(i)
            mx = 0
            mx_prod = ''
            for j in range(len(prod_list)):
                seq=difflib.SequenceMatcher(None, product,prod_list[j])
                if seq.ratio()*100>mx:
                    mx = seq.ratio()*100
                    mx_prod = prod_list[j]
            product = mx_prod
            print('')
            print(colored('Just to confirm, you purchased', 'blue'),
                  colored(''.join(product), 'blue'),colored('right?', 'blue'))
            userInput9 = input(">>> ")
            print('')
            if senti_analyzer.polarity_scores(userInput9)['compound']>0:
                print(colored('Great you are all set, thank you and have a '
                              'great day!', 'blue'))
                right_product = True
            else:
                old_prod.append(product)
