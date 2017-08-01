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

# adding a genre column
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