import os, re, io
wd = 'C:\Users\Ray\Desktop\TMProject'
os.chdir(wd)
import pandas as pd
import numpy as np

## FOR MULTIPLE FILES IN A FOLDER

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

## FOR SINGLE FILE (CSV)
import csv
source = open('C:\\Users\\Ray\\Desktop\\TMProject\\corpus_blogs.csv')
blogs = list(csv.reader(source))

df = pd.DataFrame()
df['text'] = blogs

## CLEAN - TOKENIZE - NOSTOP

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
