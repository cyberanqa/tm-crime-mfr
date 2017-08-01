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
df = pd.DataFrame.from_dict(files, orient='index')
df.columns = ['file_name', 'content']

df2 = pd.DataFrame({'file_name': files.keys(), 'content': files.values()})
