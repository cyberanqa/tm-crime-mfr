### TOPIC MODELLING ###

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

### LDA ###
    
from gensim import corpora, models
dictionary = corpora.Dictionary(var)
tok_bow = [dictionary.doc2bow(tok) for tok in var]

mdl = models.LdaModel(tok_bow, id2word=dictionary, num_topics=5, passes=20, update_every=0, alpha=None, eta=None, decay=0.5, distributed=False)

# print topics as word distributions
for i in range(5):
    print('topic', i)
    print([t[0] for t in mdl.show_topic(i, 10)])
    print('-----')

# a list of (word, probability) 2-tuples for most probable words in topic
mdl.show_topic(1, topn=10)
# topics-term matrix
topics_terms = mdl.state.get_lambda()
print topics_terms

# for future use
mdl.save('topic.model')

## Visualization

import pyLDAvis.gensim
visu = pyLDAvis.gensim.prepare(mdl, slic_bow, dictionary)

pyLDAvis.save_html(visu,'vis_sliced.html')
