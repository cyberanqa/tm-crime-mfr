# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:44:29 2017

@author: Alonso
"""
import os, re

wd = 'c:/Users/Alonso/Desktop/TMProject'
os.chdir(wd)

import textminer as tm

## files = tm.read_dir_txt('c:/Users/Alonso/Desktop/TMProject/COHA/')

def read_directory(directory):
    """
    Import multiple txt files from a directory
    """
    filenames = os.listdir(directory)
    result_list = []
    for txt in filenames:
        filepath = directory + txt
        text = tm.read_txt(filepath)
        result_list.append(text)
    return(result_list)

def read_directory_dic(directory):
    """
    Import multiple txt files from a directory in dictionary form
    """
    filenames = os.listdir(directory)
    result_list = {}
    for txt in filenames:
        filepath = directory + txt
        text = tm.read_txt(filepath)
        f = {txt: text}
        result_list.update(f)
    return(result_list)

files = read_directory('c:/Users/Alonso/Desktop/TMProject/COHA/')

###



def slice_tokens(tokens, n = 100, cut_off = True):
    """
    slice tokenized text in slices of n tokens
    - end cut off for full length normailization
    """
    # result: list of slices
    slices = []
    # slice tokens
    for i in range(0, len(tokens), n):
        slices.append(tokens[i: (i + n)])
    # cut_off function
    if cut_off:
        del slices[-1]
    return slices


####
import pandas as pd
labmt = pd.read_csv('labmt_dict.csv', sep = '\t', encoding = 'utf-8', index_col = 0)

avg = labmt.happiness_average.mean()
sent_dict = (labmt.happiness_average - avg).to_dict()
    
    
###

sentDic = []

for text in files:
    tokens = tm.tokenize(text, lentoken = 1)
    slices = slice_tokens(tokens, 50, True)
    labmt = pd.read_csv('labmt_dict.csv', sep = '\t', encoding = 'utf-8', index_col = 0)
    avg = labmt.happiness_average.mean()
    sent_dict = (labmt.happiness_average - avg).to_dict()
    sent_vects = []
    for s in slices:
        sent_vects.append(sum([sent_dict.get(token, 0.0) for token in s]))
    sentDic.append(sent_vects)
 
#• delete sentiment scores equal to zero    
del(sentDic[224])
del(sentDic[680])

sentAvg = []    
for sent in sentDic:
    sentAvg.append(sum(sent) / len(sent))

###

import quickndirty as qd
qd.plotdist(sentAvg)
qd.plotvars(sentAvg)
