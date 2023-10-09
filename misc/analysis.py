#%%
# load kedro context
%run '/home/klara/bfl-winnie/.ipython/profile_default/startup/00-kedro-init.py'
%reload_kedro

# %%

import re
import pandas as pd

# load data
messages_table = context.catalog.load("primary_messages")  # %%

# quantify message rate per day each year
messages_table['question_asked_time'].groupby([messages_table.question_asked_time.dt.year]).agg('count')/365


#%%
# merge Q&A

q = messages_table[['id','question','question_asked_time']].dropna()
a = messages_table[['id','answer','answer_given_time']].dropna()

q.columns = ['id','text','timestamp']
a.columns = ['id','text','timestamp']

qa = pd.concat([q, a])

# %%
# get some stats of raw data to begin with

import nltk

def flatten(t):
    return [item for sublist in t for item in sublist]

q_tokens = [nltk.word_tokenize(x) for x in q['text']]
q_tokens_flat = flatten(q_tokens)

a_tokens = [nltk.word_tokenize(x) for x in a['text']]
a_tokens_flat = flatten(a_tokens)

qa_tokens = [nltk.word_tokenize(x) for x in qa['text']]
qa_tokens_flat = flatten(qa_tokens)


q_tokens_flat = [token.lower() for token in q_tokens_flat]
a_tokens_flat = [token.lower() for token in a_tokens_flat]
qa_tokens_flat = [token.lower() for token in qa_tokens_flat]

#%%
print(f'There are {len(set(q_tokens_flat))} unique words in questions.')
print(f'Lexical richness in text: {len(set(q_tokens_flat))/len(q_tokens_flat) * 100} % ')

print(f'There are {len(set(a_tokens_flat))} unique words in answers.')
print(f'Lexical richness in text: {len(set(a_tokens_flat))/len(a_tokens_flat) * 100} % ')

print(f'There are {len(set(qa_tokens_flat))} unique words in questions and answers.')
print(f'Lexical richness in text: {len(set(qa_tokens_flat))/len(qa_tokens_flat) * 100} % ')


#%%

fdist_q = nltk.FreqDist(q_tokens_flat)
fdist_a = nltk.FreqDist(a_tokens_flat)
fdist_qa = nltk.FreqDist(qa_tokens_flat)

#fdist.most_common(50)
#fdist.plot(50)

unique_words_q = set(q_tokens_flat)
unique_words_a = set(a_tokens_flat)
unique_words_qa = set(qa_tokens_flat)


# %%

# spell checking

import nltk
from spellchecking import *
import numpy as np

word_list =  nltk.corpus.brown.words()#nltk.corpus.reuters.words() #nltk.corpus.brown.words() # or nltk.corpus.words.words()

def get_unusual_words(text,corpus):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in corpus)
    unusual = text_vocab - english_vocab
    return sorted(unusual)

unusual_words_q = get_unusual_words(q_tokens_flat,word_list)
unusual_words_a = get_unusual_words(a_tokens_flat,word_list)
unusual_words_qa = get_unusual_words(qa_tokens_flat,word_list)


unusual_words_q = list(set(unusual_words_q) - set(false_positives) )
unusual_words_a = list(set(unusual_words_a) - set(false_positives) )
unusual_words_qa = list(set(unusual_words_qa) - set(false_positives) )


# get frequency distribution

unusual_words_q_freq = [fdist_q.get(wordkey) for wordkey in unusual_words_q]
unusual_words_a_freq = [fdist_a.get(wordkey) for wordkey in unusual_words_a]
unusual_words_qa_freq = [fdist_qa.get(wordkey) for wordkey in unusual_words_qa]


unusual_words_q_total = np.sum(unusual_words_q_freq)
unusual_words_a_total = np.sum(unusual_words_a_freq)
unusual_words_qa_total = np.sum(unusual_words_qa_freq)

# BEFORE ANY PREPROCESSING
print(f'{(unusual_words_q_total/len(q_tokens_flat))*100} % of words in questions are not in nltk (corrected) brown corpus.') 
print(f'{(unusual_words_a_total/len(a_tokens_flat))*100} % of words in answers are not in nltk (corrected) brown corpus.') 
print(f'{(unusual_words_qa_total/len(qa_tokens_flat))*100} % of words in questions+answers are not in nltk (corrected)  brown corpus.') 

#visualise
'''
unusual_words_df =  pd.DataFrame(data = {'word':unusual_words,'freq':unusual_words_freq})
unusual_words_df_sorted = unusual_words_df.sort_values(by=['freq'],ascending=False)
unusual_words_total = unusual_words_df['freq'].sum()
import matplotlib.pyplot as plt

lower, upper = 0,100
plt.figure(figsize=(5, 20))
plt.plot(unusual_words_df_sorted['freq'][lower:upper][::-1],unusual_words_df_sorted['word'][lower:upper][::-1])
'''

#%%

# get an idea about query length

q_tokens_len = [len(x) for x in q_tokens]
a_tokens_len = [len(x) for x in a_tokens]
qa_tokens_len = [len(x) for x in qa_tokens]


# before any preprocessing
print(f'Mean length in words of questions is {np.mean(q_tokens_len)}, with median of {np.median(q_tokens_len)} and std of {np.std(q_tokens_len)}.')
print(f'Mean length in words of answers is {np.mean(a_tokens_len)}, with median of {np.median(a_tokens_len)} and with std of {np.std(a_tokens_len)}.')
print(f'Mean length in words of questions+answers is {np.mean(qa_tokens_len)}, with median of {np.median(qa_tokens_len)} and with std of {np.std(qa_tokens_len)}.')


#%%

# preprocess


from winnie3.d00_utils.preprocessing import run_preprocessing_steps


preprocessing_steps = ['clean','denoise_text','remove_luganda','replace_words','remove_punctuation','replace_contractions','remove_numbers','remove_custom_stop_words']

q_preprocessed = run_preprocessing_steps(series=q['text'], steps=preprocessing_steps)

a_preprocessed = run_preprocessing_steps(series=a['text'], steps=preprocessing_steps)


#%%

q_tokens_preprocessed = q_preprocessed.map(nltk.word_tokenize)
a_tokens_preprocessed = a_preprocessed.map(nltk.word_tokenize)

# filter by number of words
upper_limit = 200
lower_limit = 2

q_preprocessed = q_preprocessed[(lower_limit< q_tokens_preprocessed.map(len)) & (q_tokens_preprocessed.map(len) < upper_limit)]
q_preprocessed.reset_index(drop=True, inplace=True)

a_preprocessed = a_preprocessed[(lower_limit< a_tokens_preprocessed.map(len)) & (a_tokens_preprocessed.map(len) < upper_limit)]
a_preprocessed.reset_index(drop=True, inplace=True)


# %%

q_tokens_preprocessed = [nltk.word_tokenize(x) for x in q_preprocessed]
q_tokens_flat_preprocessed = flatten(q_tokens_preprocessed)

a_tokens_preprocessed = [nltk.word_tokenize(x) for x in a_preprocessed]
a_tokens_flat_preprocessed = flatten(a_tokens_preprocessed)
#%%

print(f'There are {len(set(q_tokens_flat))} unique words in questions. (before preprocessing)')

print(f'There are {len(set(q_tokens_flat_preprocessed))} unique words in questions. (after preprocessing)')

print(f'Lexical richness in text: {len(set(q_tokens_flat))/len(q_tokens_flat) * 100} % (before preprocessing)')

print(f'Lexical richness in text: {len(set(q_tokens_flat_preprocessed))/len(q_tokens_flat_preprocessed) * 100} % (after preprocessing)')

#%%

print(f'There are {len(set(a_tokens_flat))} unique words in answers. (before preprocessing)')

print(f'There are {len(set(a_tokens_flat_preprocessed))} unique words in answers. (after preprocessing)')

print(f'Lexical richness in text: {len(set(a_tokens_flat))/len(a_tokens_flat) * 100} % (before preprocessing)')

print(f'Lexical richness in text: {len(set(a_tokens_flat_preprocessed))/len(a_tokens_flat_preprocessed) * 100} % (after preprocessing)')

# spell check before and after
#%%

unusual_words_q_preprocessed = set(get_unusual_words(q_tokens_flat_preprocessed,word_list)) - set(false_positives)

unusual_words_a_preprocessed = set(get_unusual_words(a_tokens_flat_preprocessed,word_list)) - set(false_positives)


#%%

fdist = nltk.FreqDist(q_tokens_flat_preprocessed)
unusual_words_q_freq_preprocessed = [fdist.get(wordkey) for wordkey in unusual_words_q_preprocessed]

fdist = nltk.FreqDist(a_tokens_flat_preprocessed)
unusual_words_a_freq_preprocessed = [fdist.get(wordkey) for wordkey in unusual_words_a_preprocessed]

#%%

print(f'{(sum(unusual_words_q_freq)/len(q_tokens_flat))*100} % of q words are not in nltk (corrected) brown corpus. (before preprocessing)')
print(f'{(sum(unusual_words_q_freq_preprocessed)/len(q_tokens_flat_preprocessed))*100} % of q words are not in nltk (corrected) brown corpus. (after preprocessing)') 

print(f'{(sum(unusual_words_a_freq)/len(a_tokens_flat))*100} % of a words are not in nltk (corrected) brown corpus. (before preprocessing)')
print(f'{(sum(unusual_words_a_freq_preprocessed)/len(a_tokens_flat_preprocessed))*100} % of a words are not in nltk (corrected) brown corpus. (after preprocessing)') 

#%%
# visualise
'''
import matplotlib.pyplot as plt

unusual_words_df =  pd.DataFrame(data = {'word':list(unusual_words_preprocessed),'freq':list(unusual_words_freq_preprocessed)})
unusual_words_df_sorted = unusual_words_df.sort_values(by=['freq'],ascending=False)
lower, upper = 0,100
plt.figure(figsize=(5, 20))
plt.plot(unusual_words_df_sorted['freq'][lower:upper][::-1],unusual_words_df_sorted['word'][lower:upper][::-1])
'''


len(q_preprocessed.value_counts()) # number of unique questions (without any merging)
len(q_preprocessed.value_counts())/len(q_preprocessed) # number of unique questions (proportion)


len(a_preprocessed.value_counts()) # number of unique questions (without any merging)
len(a_preprocessed.value_counts())/len(a_preprocessed) # number of unique questions (proportion)



# less answers than questions, makes sense!


def find_queries(text):
    pattern = re.compile(r"what is \S*\s?")
    return bool(re.match(pattern,text))

# %%




