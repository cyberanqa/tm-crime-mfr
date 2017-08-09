# -*- coding: utf-8 -*-
"""
Text Mining
COCA
"""
import os, re, io
wd = 'C:\Users\Ray\Desktop\TMProject'
os.chdir(wd)
import pandas as pd
import numpy as np

filepath = 'C:\Users\Ray\Desktop\TMProject\COCA'
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

# genre column
genre = []
for text in filenames:
    text = re.sub(r'[^a-zA-Z]','',text)
    text = text.rstrip()
    text = re.sub(r'[txt]','',text)
    text = text[1:5]
    genre.append(text)
df['genre'] = genre

# year column
year = []
for text in filenames:
    text = re.sub(r'[a-zA-Z]','',text)
    text = text[2:6]
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
mdl.save('sliced.model')

for i in range(5):
    print('topic', i)
    print([t[0] for t in mdl.show_topic(i, 10)])
    print('-----')

# a list of (word, probability) 2-tuples for most probable words in topic
mdl.show_topic(1, topn=10)

'''
('topic', 0)
[u'qwq', u'spencer', u'police', u'gray', u'john', u'war', u'foreman', u'gibson', u'michael', u'children']
-----
('topic', 1)
[u'players', u'team', u'season', u'smith', u'free', u'states', u'against', u'since', u'academic', u'committee']
-----
('topic', 2)
[u'moriarty', u'against', u'human', u'law', u'mckinney', u'killed', u'unidentified', u'old', u'next', u'same']
-----
('topic', 3)
[u'season', u'record', u'team', u'left', u'florida', u'offense', u'five', u'end', u'four', u'why']
-----
('topic', 4)
[u'reilly', u'york', u'american', u'government', u'crime', u'abuse', u'federal', u'united', u'states', u'against']

'''

## Visualization

import pyLDAvis.gensim
visu = pyLDAvis.gensim.prepare(mdl, slic_bow, dictionary)

pyLDAvis.save_html(visu,'vis_sliced.html')

### LSA ###

import gensim
lsi = gensim.models.lsimodel.LsiModel(slic_bow, id2word=dictionary, num_topics=5)
lsi.print_topics()

'''
[(0,
  u'0.099*"president" + 0.098*"against" + 0.098*"york" + 0.090*"season" + 0.089*"since" + 0.088*"qwq" + 0.086*"why" + 0.086*"killed" + 0.085*"crime" + 0.084*"team"'),
 (1,
  u'0.276*"season" + 0.192*"quarterback" + 0.192*"record" + 0.173*"offense" + 0.167*"defense" + 0.161*"team" + 0.133*"defensive" + 0.129*"players" + 0.124*"offensive" + -0.114*"gargano"'),
 (2,
  u'-0.362*"reilly" + -0.240*"qwq" + 0.146*"gargano" + -0.143*"morones" + 0.128*"officials" + 0.125*"york" + 0.124*"chulpayev" + 0.124*"federal" + 0.123*"company" + -0.101*"moriarty"'),
 (3,
  u'0.577*"reilly" + -0.328*"qwq" + 0.237*"morones" + -0.128*"moriarty" + 0.128*"border" + -0.099*"gibson" + 0.094*"animals" + 0.087*"begala" + 0.087*"rivera" + 0.075*"rights"'),
 (4,
  u'0.255*"qwq" + -0.170*"gayle" + -0.167*"king" + -0.131*"richard" + -0.126*"keteyian" + 0.123*"smith" + -0.118*"schlesinge" + 0.115*"war" + -0.110*"erica" + -0.108*"insurance"')]
'''

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
lsidf.to_csv('lsi_coca.csv')
