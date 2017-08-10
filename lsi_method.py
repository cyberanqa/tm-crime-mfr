## preparation of texts

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

# create an object for dictionary
var = []
for l in cri:
    var.append([item for sublist in l for item in sublist])

### LSI ###

import gensim
from gensim import corpora, models
dictionary = corpora.Dictionary(var)
tok_bow = [dictionary.doc2bow(tok) for tok in var]

# term-document matrix decomposition, vector representation of document
lsi = gensim.models.lsimodel.LsiModel(tok_bow, id2word=dictionary, num_topics=5)
# print the most contributing words for each topic
lsi.print_topics()
lsi.show_topics(formatted=False)

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
