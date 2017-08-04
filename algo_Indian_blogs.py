# -*- coding: utf-8 -*-
"""
TMProject
draft
"""
import os, re
import pandas as pd
import numpy as np
wd = 'C:\Users\Ray\Desktop\TMProject'
os.chdir(wd)

''' IMPORT AND CLEAN TEXT '''
import csv
source = open('C:\\Users\\Ray\\Desktop\\TMProject\\corpus_blogs.csv')
blogs = list(csv.reader(source))

df = pd.DataFrame()
df['raw_text'] = blogs

# clean text (also without commas or dots)
textClean = []
for text in df['raw_text']:
    text = str(text)
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = re.sub(r' +',' ',text)
    text = text.rstrip()
    textClean.append(text)
df['clean_text'] = textClean

''' TOKENIZE AND REMOVE STOPWORDS '''

# tokenize clean text + lower-casing
def tokenize(input, length = 0, casefold = False):
    tokenizer = re.compile(r'[^A-Za-z]+')
    if casefold:
        input = input.lower()
    tokens = [token for token in tokenizer.split(input) if len(token) > length]
    return tokens
token_text = [tokenize(i, 1, True) for i in textClean]
df['token_text'] = token_text

# generate stopword list from unsliced text
from collections import defaultdict
from operator import itemgetter

def gen_ls_stoplist(input, n = 100):
    t_f_total = defaultdict(int)
    for text in input:
        for token in text:
            t_f_total[token] += 1
    nmax = sorted( t_f_total.items(), key = itemgetter(1), reverse = True)[:n]
    return [elem[0] for elem in nmax]
sw = gen_ls_stoplist(token_text, 150)

# apply stopword list
token_nostop = []
no_sw = []
for fragment in token_text:
    no_sw = [token for token in fragment if token not in sw]
    token_nostop.append(no_sw)
df['token_nostop'] = token_nostop
   
''' CRIME DICT '''
import io
cri_dic = io.open('C:\Users\Ray\Desktop\TMProject\crime_dic.txt', 'r', encoding = 'utf-8').read()
cri_dic_clean = re.sub(r'[^a-zA-Z]',' ',cri_dic)
cri_dic_clean = re.sub(r' +',' ', cri_dic_clean)
cri_dic_clean = cri_dic_clean.rstrip()
token_cri_dic = tokenize(cri_dic_clean, True)

# filter the texts by crime dictionary
crime = []
for item in token_nostop:
    text_res = []
    for w in item:
        if w in token_cri_dic:
            text_res.append(item)
    crime.append(text_res)
df['crime'] = crime

# delete empty strings
cri = list(filter(None, crime))

#  
var = cri

for i in range(0, len(cri)):
    var = []
    for n in range(0, len(cri[i])):
        var = var + cri[i][n]

var = []
for l in cri:
    var.append([item for sublist in l for item in sublist])
        
var[0]
flat_list = [item for sublist in var[0] for item in sublist]
    
''' TOPIC MODELLING - LDA'''

from __future__ import division
import numpy as np
from gensim import corpora, models

# distribution over topics
dictionary = corpora.Dictionary(var)
print(dictionary.num_docs)   
print(dictionary.keys())    
print(dictionary.values())

tok_bow = [dictionary.doc2bow(tok) for tok in var]

# train the model
mdl = models.LdaModel(tok_bow, id2word = dictionary, num_topics = 15, random_state = 1234)

# you can also run on multiple cores in your computer (distributed=False)
# mdl = models.LdaModel(tok_bow, id2word=dictionary, num_topics=k, passes=25, update_every=0, alpha=None, eta=None, decay=0.5, distributed=False)

for i in range(15):
    print('topic', i)
    # list of (word, probability) for most probable words in topic
    print[t[0] for t in mdl.show_topic(i, 15)]
    print('-----')

# test model

query = open('C:\\Users\\Ray\\Desktop\\TMProject\\test_article_crime.txt').read()
query = query.lower().split() # string into list
vocab = dictionary.values() # dict for new document based on our data
query = [w for w in query if w in vocab]

query = dictionary.doc2bow(query)

mdl[query] # train model on new data
mdl.print_topics(20,10)

# KL divergence
def get_theta(tok_bow, mdl):
   tmp = mdl.get_document_topics(tok_bow, minimum_probability=0)
   return [p[1] for p in tmp]

def kl_div(p, q):
   p = np.asarray(p, dtype=np.float)
   q = np.asarray(q, dtype=np.float)
   return np.sum(np.where(p != 0,(p-q) * np.log10(p / q), 0))

tok1 = tok_bow[1]
tok2 = tok_bow[-100]
tok3 = tok_bow[2]

# difference between topics: how many times
print kl_div(get_theta(tok1, mdl), get_theta(tok2, mdl))
print kl_div(get_theta(tok1, mdl), get_theta(tok3, mdl))
# probability distribution for each topic
print get_theta(tok1, mdl)
