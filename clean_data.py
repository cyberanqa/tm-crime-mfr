# -*- coding: utf-8 -*-
import os, re
import pandas as pd

wd = 'c:/Users/Alonso/Desktop/TMProject'
os.chdir(wd)

import textminer as tm

## files = tm.read_dir_txt('c:/Users/Alonso/Desktop/TMProject/COHA/')

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

### df - create a data frame with pandas

df = pd.DataFrame({'file_name': files.keys(), 'content': files.values()})
