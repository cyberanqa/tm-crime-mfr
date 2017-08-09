# -*- coding: utf-8 -*-
"""
Text Mining
State of the Union addresses
"""
import os, re, io
wd = 'C:\Users\Ray\Desktop\TMProject'
os.chdir(wd)
import pandas as pd
import numpy as np

filepath = 'C:\Users\Ray\Desktop\TMProject\state_union'
filenames = os.listdir(filepath)
text_list = []
for filename in filenames:
    tmp = filepath + '\\' + filename
    with io.open(tmp, 'r', encoding = 'utf-8') as f:
        text = f.read()
    text_list.append(text)

df = pd.DataFrame()
df['filename'] = filenames
df['text'] = text_list

# president column
pres = []
for text in filenames:
    text = re.sub(r'[^a-zA-Z]','',text)
    text = text.rstrip()
    text = re.sub(r'[txt]','',text)
    pres.append(text)
df['pres'] = pres

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

# tokenize
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
    slices = []
    for i in range(0, len(texts), n):
        slices.append(texts[i: (i + n)])
    if cut_off:
        del slices[-1]
    return slices

token_sliced = [slice_text(i, 100) for i in token_nostop]
df['token_sliced'] = token_sliced

# crime dictionary
with io.open('C:\Users\Ray\Desktop\TMProject\crime_dic_all.txt', 'r', encoding = 'utf-8') as f:
    cri_dic = f.read()

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

# create an object for dictionary
var = []
for l in cri:
    var.append([item for sublist in l for item in sublist])
        
### TOPIC MODELLING - LDA ###

from gensim import corpora, models
dictionary = corpora.Dictionary(var)
slic_bow = [dictionary.doc2bow(slic) for slic in var]

mdl = models.LdaModel(slic_bow, id2word=dictionary, num_topics=5, passes=20, update_every=0, alpha=None, eta=None, decay=0.5, distributed=False)
mdl.save('topic_speech.model')

for i in range(5):
    print('topic', i)
    print([t[0] for t in mdl.show_topic(i, 10)])
    print('-----')

'''
('topic', 0)
[u'terrorists', u'fight', u'citizens', u'free', u'peace', u'against', u'terror', u'hope', u'yet', u'men']
-----
('topic', 1)
[u'crime', u'why', u'community', u'she', u'use', u'which', u'go', u'bill', u'protect', u'system']
-----
('topic', 2)
[u'weapons', u'iraq', u'could', u'money', u'inspectors', u'nuclear', u'recovery', u'drugs', u'use', u'which']
-----
('topic', 3)
[u'iraq', u'terrorists', u'law', u'weapons', u'nations', u'enforcement', u'against', u'women', u'regime', u'act']
-----
('topic', 4)
[u'crime', u'community', u'thank', u'challenge', u'federal', u'weapons', u'citizens', u'state', u'police', u'members']
'''

## visualization

import pyLDAvis.gensim
visu = pyLDAvis.gensim.prepare(mdl, slic_bow, dictionary)

pyLDAvis.save_html(visu,'vis_speech.html')

## UPDATE former model on new documents

from gensim.models import LdaModel
coca_model = LdaModel.load('sliced.model')

coca_model.update(slic_bow)
for i in range(5):
    print('topic', i)
    print([t[0] for t in coca_model.show_topic(i, 10)])
    print('-----')

'''
('topic', 0)
[u'investigations', u'psyche', u'overestimate', u'australian', u'abstract', u'confident', u'speed', u'vehicle', u'crash', u'sirens']
-----
('topic', 1)
[u'harm', u'foundation', u'coercion', u'retention', u'interviewed', u'rational', u'discredit', u'next', u'press', u'surface']
-----
('topic', 2)
[u'imply', u'defines', u'develops', u'paula', u'goddesses', u'dances', u'cosmos', u'narratives', u'layers', u'womanhood']
-----
('topic', 3)
[u'season', u'record', u'team', u'left', u'florida', u'offense', u'five', u'end', u'four', u'why']
-----
('topic', 4)
[u'socialist', u'espouse', u'resources', u'document', u'conceived', u'finds', u'cynicism', u'stance', u'behind', u'members']
'''

### LSA ###

import gensim
lsi = gensim.models.lsimodel.LsiModel(slic_bow, id2word=dictionary, num_topics=5)
lsi.print_topics()

## visualize words and percentages
lex = lsi.show_topics(5, formatted=False)
lexi = str(lex)
lexic = lexi.split(',')

# select percentages into one column
lex_num = []
for text in lexic:
    text = re.sub(r'[a-zA-Z]','',text)
    text = re.sub(r'[^\d|.?]','',text)
    lex_num.append(text)
lex_num = filter(None, lex_num)
indexes = [0, 11, 22, 33, 44]
for index in sorted(indexes, reverse = True):
    del lex_num[index]
# select words into one column
lex_word = []
for text in lexic:
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = re.sub(r' +',' ', text)
    lex_word.append(text)
lexi_word = ' '.join(lex_word).split()
del lexi_word[::2]

# create dataframe with two columns    
lsidf = pd.DataFrame()
lsidf['percent'] = lex_num
lsidf['words'] = lexi_word
# into csv
lsidf.to_csv('lsi.csv')

'''
[(0,
  u'0.222*"crime" + 0.163*"weapons" + 0.124*"community" + 0.109*"terrorists" + 0.104*"law" + 0.104*"peace" + 0.104*"which" + 0.104*"nuclear" + 0.102*"against" + 0.100*"iraq"'),
 (1,
  u'-0.271*"crime" + 0.224*"iraq" + 0.206*"weapons" + 0.156*"terrorists" + -0.140*"community" + 0.138*"terror" + 0.136*"nations" + 0.125*"iraqi" + 0.123*"regime" + -0.108*"police"'),
 (2,
  u'0.303*"weapons" + 0.151*"iraq" + 0.146*"inspectors" + 0.128*"saddam" + 0.122*"hussein" + 0.113*"nuclear" + -0.112*"fight" + 0.110*"crime" + 0.093*"money" + -0.089*"which"'),
 (3,
  u'0.195*"gun" + 0.153*"thank" + 0.144*"challenge" + -0.140*"something" + -0.119*"crime" + 0.112*"state" + -0.110*"secure" + -0.107*"system" + 0.107*"weapons" + 0.106*"parents"'),
 (4,
  u'-0.310*"challenge" + 0.211*"gun" + 0.165*"which" + -0.148*"peace" + -0.137*"federal" + -0.112*"helping" + -0.112*"crime" + -0.110*"proud" + -0.098*"percent" + -0.086*"schools"')]
 '''
