#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 18:57:27 2018

@author: ambikaajaggi
"""
import os
import pandas as pd
import pickle
from termcolor import colored

cwd = os.getcwd()  # Gets the current working directory (cwd)
files = os.listdir(cwd)  # Gets all the files in that directory
old_df=pd.read_csv('GrammarandProductReviews.csv') # previous datafile
new_df = pd.read_csv('GrammarandProductReviews copy.csv') # new datafile

# =============================================================================
# Create functions that will calculated user requested data
# =============================================================================

def fake_reviews(inpt):
    count = 0
    n = len(inpt)
    for i in range(n):
        if (inpt['reviews.didPurchase'][i]==False):
            count+=1
    return count

def general_sentiment(inpt):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid
    count = []
    senti_analyzer = sid()
    n = len(inpt)
    for i in range(n):
        count.append(senti_analyzer.polarity_scores(
                str(inpt['reviews.text'][i]))['compound'])
    return (sum(count))/n

def not_empty_fields(inpt):
    return inpt.notnull().sum()

def average_rating(inpt):
    count = 0
    n = len(inpt)
    for i in range(n):
        count+=inpt['reviews.rating'][i]
    return count/n

def improved_rating_product(inpt1, inpt2):
    list_of_products = []
    if len(inpt2['name'].unique())>len(inpt1['name'].unique()):
        temp = inpt1
        inpt1 = inpt2
        inpt2 = temp
    for i in range(len(inpt2)):
        if inpt2['name'][i] not in list_of_products:
            list_of_products.append(inpt2['name'][i])
            
    improved = {}
    for prod in list_of_products:
        inpt1_new = inpt1.loc[inpt1['name']==prod]
        inpt2_new = inpt2.loc[inpt2['name']==prod]
        rating_inpt1_new = (inpt1_new['reviews.rating'].sum())/len(inpt1_new)
        rating_inpt2_new = (inpt2_new['reviews.rating'].sum())/len(inpt2_new)
        rating3 = ((inpt1_new['reviews.rating'].sum()+
                    inpt2_new['reviews.rating'].sum())/
                      (len(inpt1_new)+len(inpt2_new)))
        
        improved[prod] = ['improved', rating_inpt1_new<rating_inpt2_new, 'old',  
                rating_inpt1_new, 'new', rating_inpt2_new, 'total', rating3]
    return improved

def improved_rating_brand(inpt1, inpt2):
    list_of_brands = []
    if len(inpt2['brand'].unique())>len(inpt1['brand'].unique()):
        temp = inpt1
        inpt1 = inpt2
        inpt2 = temp
    for i in range(len(inpt2)):
        if inpt2['brand'][i] not in list_of_brands:
            list_of_brands.append(inpt2['brand'][i])
            
    improved = {}
    for brand in list_of_brands:
        inpt1_new = inpt1.loc[inpt1['brand']==brand]
        inpt2_new = inpt2.loc[inpt2['brand']==brand]
        rating_inpt1_new = (inpt1_new['reviews.rating'].sum())/len(inpt1_new)
        rating_inpt2_new = (inpt2_new['reviews.rating'].sum())/len(inpt2_new)
        rating3 = ((inpt1_new['reviews.rating'].sum()+
                    inpt2_new['reviews.rating'].sum())/
                      (len(inpt1_new)+len(inpt2_new)))
        
        improved[brand] = ['improved',rating_inpt1_new<rating_inpt2_new, 'old',  
                rating_inpt1_new, 'new', rating_inpt2_new, 'total', rating3]
    return improved

def improved_rating_manufacturer(inpt1, inpt2):
    list_of_manufacturer = []
    if len(inpt2['manufacturer'].unique())>len(inpt1['manufacturer'].unique()):
        temp = inpt1
        inpt1 = inpt2
        inpt2 = temp
    for i in range(len(inpt2)):
        if inpt2['manufacturer'][i] not in list_of_manufacturer:
            list_of_manufacturer.append(inpt2['manufacturer'][i])
            
    improved = {}
    for manufacturer in list_of_manufacturer:
        inpt1_new = inpt1.loc[inpt1['manufacturer']==manufacturer]
        inpt2_new = inpt2.loc[inpt2['manufacturer']==manufacturer]
        rating_inpt1_new = (inpt1_new['reviews.rating'].sum())/len(inpt1_new)
        rating_inpt2_new = (inpt2_new['reviews.rating'].sum())/len(inpt2_new)
        rating3 = ((inpt1_new['reviews.rating'].sum()+
                    inpt2_new['reviews.rating'].sum())/
                      (len(inpt1_new)+len(inpt2_new)))
        
        improved[manufacturer] = ['improved',rating_inpt1_new<rating_inpt2_new, 
                              'old', rating_inpt1_new, 'new', rating_inpt2_new, 
                                                              'total', rating3]
    return improved

def promotion_product(inpt1, inpt2):
    list_of_products = []
    if len(inpt2['name'].unique())>len(inpt1['name'].unique()):
        temp = inpt1
        inpt1 = inpt2
        inpt2 = temp
    for i in range(len(inpt2)):
        if inpt2['name'][i] not in list_of_products:
            list_of_products.append(inpt2['name'][i])
            
    promotion = {}
    pattern = "This review was collected as part of a promotion"
    for prod in list_of_products:
        inpt1_new = inpt1.loc[inpt1['name']==prod]
        inpt2_new = inpt2.loc[inpt2['name']==prod]
        
        promo1 = inpt1_new['reviews.text'].str.contains(pattern)
        promo2 = inpt2_new['reviews.text'].str.contains(pattern)
        
        count1 = 0
        for i in promo1:
            if i == 1:
                count1+=1
        
        count2 = 0
        for i in promo2:
            if i == 1:
                count1+=2
        promotion[prod] = ['old', str(count1)+'/'+str(len(promo1)), 'new', 
                 str(count2)+'/'+str(len(promo2)), 'total', 
                 str(count1+count2)+'/'+str((len(promo1)+len(promo2)))]
    return promotion

def best_brand(inpt1, inpt2, inpt3):
    df_total = inpt1.append(inpt2,  ignore_index=True)
    dict_of_brands = {}
    if inpt3 != []:
        for i in range(len(df_total)):
            if inpt3 in (df_total['name'][i]).lower():
                if df_total['name'][i] not in dict_of_brands.keys():
                    dict_of_brands[df_total['name'][i]
                                            ] =  df_total['reviews.rating'][i]
                else:
                    dict_of_brands[df_total['name'][i]
                                            ] += df_total['reviews.rating'][i]
    for key, val in dict_of_brands.items():
        count = 0
        for i in range(len(df_total)):
            if df_total['name'][i] == key:
                count += 1
        val /= count
        dict_of_brands[key] = val
    
    best_rating = max(dict_of_brands.values())
    
    best_brands = []
    for key in dict_of_brands.keys():
        if dict_of_brands[key] == best_rating:
            best_brands.append(key)
            
    return best_brands

# =============================================================================
# Create dictionary of key-words and corresponding function calls, saving all 
# dictionary computations into a pickle file
# =============================================================================

dc = {'previous fake customers': fake_reviews(old_df),
      'new fake customers': fake_reviews(new_df),
      'previous average sentiment': general_sentiment(old_df),
      'new average sentiment': general_sentiment(new_df),
      'previous not empty fields': not_empty_fields(old_df),
      'new not empty fields': not_empty_fields(new_df),
      'previous average rating': average_rating(old_df),
      'new average rating': average_rating(new_df),
      'change average rating product': improved_rating_product(old_df, new_df),
      'change average rating brand': improved_rating_brand(old_df, new_df),
      'change average rating manufacturer': 
          improved_rating_manufacturer(old_df, new_df),
      'change number collected as part of promotion product': 
          promotion_product(old_df, new_df),
      'best brand for': '',
      'which should buy': ''}

save_question_data = open("Chatbot-Questions.pickle","wb")
pickle.dump(dc, save_question_data)
save_question_data.close()

import difflib
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

# =============================================================================
# Import pickles questions and create question and answer lsist that can be 
# called more easily upon
# =============================================================================

question_data_f = open("Chatbot-Questions.pickle","rb")
dc = pickle.load(question_data_f)
question_data_f.close()

key_list = list(dc.keys())
value_list = list(dc.values())

# =============================================================================
# Create the question asking loop that continues to prompt till the user 
# terminates it
# =============================================================================

print('')
print(colored('Ask questions about the review statistics, type end to ' 
              'terminate', 'red'))
question = input('>>> ')
while question.lower()!='end':
    mx = 0
    right_answer = ''
    right_q = ''
    for i in range(len(dc)):
        test = (difflib.SequenceMatcher(None, key_list[i],
                                        question.lower())).ratio()
        if test>mx:
            mx = test
            right_answer = value_list[i]
            right_q = key_list[i]
    
    # more compuation required if searching for a specific brand
    if right_q == 'best brand for':
        text = nltk.tokenize.word_tokenize(question)
        tags = nltk.pos_tag(text)
        indx = text.index('best')
        brand_type = ''
        for i in range(len(text)):
            if (i>indx and tags[i][1] == ('NN' or 'NNP' or 'NNPS' or 'NNS') 
                                                    and tags[i][0]!='brand'
                                                    and tags[i][0]!='brands'):
                brand_type = tags[i][0]
        Lem = WordNetLemmatizer()
        find = Lem.lemmatize(brand_type)
        right_answer = best_brand(old_df, new_df, find)
    elif right_q == 'which should buy':
        text = nltk.tokenize.word_tokenize(question)
        tags = nltk.pos_tag(text)
        indx = text.index('which')
        brand_type = ''
        for i in range(len(text)):
            if (i>indx and tags[i][1] == ('NN' or 'NNP' or 'NNPS' or 'NNS') 
                                                    and tags[i][0]!='brand'
                                                    and tags[i][0]!='brands'):
                brand_type = tags[i][0]
        Lem = WordNetLemmatizer()
        find = Lem.lemmatize(brand_type)
        right_answer = best_brand(old_df, new_df, find)
    print('')
    print(right_answer)
    print('')
    print(colored('Next question', 'red'))
    question = input('>>> ') 
print('')

# =============================================================================
# Prints the summary of the toatl of the two datasets which will become the new
# previous/old data
# =============================================================================
print(colored('Total Data Summary:', 'red'))
df_total = old_df.append(new_df,  ignore_index=True)

print('Total Fake reviews:', fake_reviews(df_total))
print('Total average sentiment:', general_sentiment(df_total))
print('Total non-empty fields:')
print(not_empty_fields(df_total))
print('Total average rating:',average_rating(df_total))