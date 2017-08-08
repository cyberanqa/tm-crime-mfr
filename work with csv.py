# -*- coding: utf-8 -*-

import os
wd = 'C:\Users\Ray\Desktop\TMProject'
os.chdir(wd)

import csv
source = open('C:\\Users\\Ray\\Desktop\\TMProject\\corpus_blogs.csv')
blogs = list(csv.reader(source))

df = pd.DataFrame()
df['raw_text'] = blogs
''
