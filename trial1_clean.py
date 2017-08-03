# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:02:50 2017

@author: Alonso
"""

import os, re, io

wd = 'c:/Users/Alonso/Desktop/TMProject'
os.chdir(wd)

import pandas as pd


###

filepath = 'c:/Users/Alonso/Desktop/TMProject/COHA/'
filenames = os.listdir(filepath)
text_list = []
for filename in filenames:
    tmp = filepath + filename
    with io.open(tmp, 'r', encoding = 'utf-8') as f:
        text = f.read()
    text_list.append(text)

df = pd.DataFrame()
df['filename'] = filenames
df['text'] = text_list

# adding a genre column (fic: fiction, mag: magazines, news: news, nf: non-fiction)
genre = []
for text in filenames:
    text = re.sub(r'[^a-zA-Z]','',text)
    text = text.rstrip()
    text = re.sub(r'[txt]','',text)
    genre.append(text)
df['genre'] = genre

# year column
year = []
for text in filenames:
    text = re.sub(r'[a-zA-Z]','',text)
    text = text[1:5]
    year.append(text)
df['year'] = year

# clean text (also without commas or dots)
textClean = []
for text in df['text']:
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = re.sub(r' +',' ', text)
    text = text.rstrip()
    textClean.append(text)
df['clean_text'] = textClean
    
"""
- First try for topic analysis
"""

# tokenize
def tokenize(input, length = 0, casefold = False):
    """
    string tokenization and lower-casing for text string
    parameters:
        - text: string to be tokenized
        - lentoken: ignore tokens shorter than or equal to lentoken (default: 0)
    """
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
    """
    generate stopword list from list of tokenized text strings
    """
    t_f_total = defaultdict(int)
    #n = 100
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
   
# sliced tokens (consider whether to include the cut off or not)
def slice_text(texts, n = 100, cut_off = True):
    """
    slice tokenized text in slices of n tokens
    - end cut off for full length normailization
    """
    # result: list of slices
    slices = []
    # slice tokens
    for i in range(0, len(texts), n):
        slices.append(texts[i: (i + n)])
    # cut_off function
    if cut_off:
        del slices[-1]
    return slices

token_sliced = [slice_text(i, 150) for i in token_nostop]
df['token_sliced'] = token_sliced

# crime dictionary

with io.open('c:/Users/Alonso/Desktop/TMProject/crime_dic.txt', 'r', encoding = 'utf-8') as f:
    cri_dic = f.read()

cri_dic_clean = re.sub(r'[^a-zA-Z]',' ',cri_dic)
cri_dic_clean = re.sub(r' +',' ', cri_dic_clean)
cri_dic_clean = cri_dic_clean.rstrip()

token_cri_dic = tokenize(cri_dic_clean, True)

# variable for the crime dictionary
multillist = list(df['token_sliced'])
cri = []
for text in multillist:
    text_res = []
    for slice in text:
        idx = 0
        for w in slice:
            if w in token_cri_dic:
                idx += 1
        if idx > 0:
            text_res.append(slice)
    cri.append(text_res)
            
df['cri'] = cri
      
# 
var = cri
var = [].join(cri)

for i in range(0, len(cri)):
    var = []
    for n in range(0, len(cri[i])):
        var = var + cri[i][n]

var = []
for l in cri:
    var.append([item for sublist in l for item in sublist])
        
var[0]
flat_list = [item for sublist in var[0] for item in sublist]
# topic modelling
### let the topic modelling begin
from gensim import corpora, models

dictionary = corpora.Dictionary(var)

###
# some of this dont do shit
print(dictionary.num_docs)    
print(dictionary.items())
print(dictionary.keys())    
print(dictionary.values())    
print(dictionary.dfs) 
###

slic_bow = [dictionary.doc2bow(slic) for slic in var]


# train the model
mdl = models.LdaModel(slic_bow, id2word = dictionary, num_topics = 5, random_state = 1234)


# explore the model
# print topics as word distributions
for i in range(5):
    print('topic', i)
    print([t[0] for t in mdl.show_topic(i, 10)])
    print('-----')


"""
vocab = dictionary.values()
query = []
for fragment in token_sliced:
    tmp = [w for w in token_sliced if w in vocab]
    query.append(tmp)

query = [w for w in token_sliced if w in vocab] 


print(query)
query = dictionary.doc2bow(query)
print(query)
mdl[query]
"""





        














