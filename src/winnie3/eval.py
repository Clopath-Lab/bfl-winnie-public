#%%
# load kedro context
%run '/home/klara/bfl-winnie/.ipython/profile_default/startup/00-kedro-init.py'
%reload_kedro

# %%

from winnie3.d00_utils.preprocessing import run_preprocessing_steps
import re
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import torch
import pickle
import difflib
import wmd as wmdlib
from collections import Counter, defaultdict
from wmd import WMD
import itertools
import random


#%%

class SpacyEmbeddings(object):
    def __getitem__(self, item):
        return nlp.vocab[item].vector

def filter_outliers(data, max_num_words=200):
    """ Filter outlier question-answer pairs from training data
    :param data: Training data 
    :param max_num_words: Threshold for number of words in outlier question-answer pairs
    :return: Filtered training data to only contain question-answer pairs with < max_num_words words.
    """

    data['num_words_ques'] = [len(re.findall(r'\w+', ques))
                              for ques in data['question']]
    data['num_words_ans'] = [len(re.findall(r'\w+', ans))
                             for ans in data['answer']]
    data = data.loc[(data['num_words_ques'] < max_num_words) &
                    (data['num_words_ans'] < max_num_words)]
    data.drop(['num_words_ques', 'num_words_ans'], axis=1)
    data.reset_index(drop=True, inplace=True)
    return data

def get_vector_faster(x):
    return nlp.make_doc(x).vector

def get_vector(x):
    return nlp(x).vector

def normalized_dot_product(a, b):
    """Calculates the normalized dot product between two numpy arrays, after
    flattening them."""

    a = torch.as_tensor(a)
    b = torch.as_tensor(b)

    a_norm = torch.linalg.norm(a)
    b_norm = torch.linalg.norm(b)

    if a_norm > 0 and b_norm > 0:
        return torch.dot(a.flatten(), b.flatten()) / (a_norm * b_norm)
    else:
        return 0

#%%


messages_table = context.catalog.load("primary_messages")  

train_data = messages_table[['id','question','answer']]

train_data = train_data.dropna(subset=['question', 'answer']) 
train_data.reset_index(drop=True, inplace=True)

train_data = filter_outliers(data=train_data)

#%%

question = train_data['question']
answer = train_data['answer']

text_dataframe = pd.DataFrame({'question': question,
                               'answer': answer})


#%%

preprocessing_steps = ['clean_blasts','denoise_text','replace_words','replace_contractions','remove_punctuation','replace_numbers_with_words','remove_punctuation','remove_custom_stop_words']
embeddings = ['tfidf','glove','glove_wmd','use']
num_neighbors = 5
train_params= {'max_df': 0.9, 'min_df': 0.01}


#%%

q_preprocessed = run_preprocessing_steps(series=question, steps=preprocessing_steps)
text_dataframe['answer'] = text_dataframe['answer'].apply(lambda x: x.replace('To respond, type LAW, type your response and send to 6115.',''))
a_preprocessed = run_preprocessing_steps(series=answer, steps=preprocessing_steps)


#%%

# load eval dataset

with open("/datadrive/eval/questions.pkl", "rb") as handle:
    question_series = pickle.load(handle)


with open("/datadrive/eval/answers.pkl", "rb") as handle:
    answer_series = pickle.load(handle)


queries_preprocessed = run_preprocessing_steps(
    series=question_series, steps=preprocessing_steps
)
queries_answers_preprocessed = run_preprocessing_steps(
    series=answer_series, steps=preprocessing_steps
)

#%%

# subset
ss_a_indices = random.sample(list(range(len(a_preprocessed))),100)

#%%

recommendations = defaultdict(list)

#%%



for emb in embeddings:
    print(emb)
    if emb=='tfidf':
        model = TfidfVectorizer(**train_params)
        corpus = q_preprocessed.values.tolist()
        model.fit(corpus)
        feature_matrix = model.transform(corpus).todense()
        query_vector = model.transform(queries_preprocessed.values.tolist()).todense()
        # for analysis
        a_model = TfidfVectorizer(**train_params)
        a_corpus = a_preprocessed.values.tolist()
        a_model.fit(a_corpus)
        a_vectors = a_model.transform(a_corpus).todense()
        c_a_vectors = a_model.transform(queries_answers_preprocessed.values.tolist()).todense()
    else:
        # spacy
        if 'glove' in emb:
            nlp = spacy.load('en_core_web_md')
            wmd_instance = wmdlib.WMD.SpacySimilarityHook(nlp)
        elif emb=='use':
            nlp = spacy.load('en_use_md')
        docs = list(nlp.pipe(q_preprocessed.tolist()))
        feature_matrix = np.stack([x.vector for x in docs])
        query_docs = list(nlp.pipe(queries_preprocessed.tolist()))
        query_vector = [q.vector for q in query_docs]
        # for analysis
        a_docs = list(nlp.pipe(a_preprocessed.tolist()))
        a_vectors = np.stack([x.vector for x in a_docs])
        c_a_docs = list(nlp.pipe(queries_answers_preprocessed.tolist()))
        c_a_vectors = np.stack([x.vector for x in c_a_docs])
        else:
            print('Embedding not implemented!')
            break 
    if 'wmd' not in emb:
        feature_dataframe = pd.DataFrame(feature_matrix, index=q_preprocessed.index)
        nearest_neighbor_model = NearestNeighbors(
            n_neighbors=num_neighbors, metric="cosine"
        )
        nearest_neighbor_model.fit(feature_dataframe)
        dist, ind = nearest_neighbor_model.kneighbors(query_vector)
    else:
        documents = {}
        for i in range(len(docs)):
            text = docs[i]
            tokens = [t for t in text if t.is_alpha and not t.is_stop]
            words = Counter(t.text for t in tokens)
            orths = {t.text: t.orth for t in tokens}
            sorted_words = sorted(words)
            documents[i] = (i, [orths[t] for t in sorted_words],
                                np.array([words[t] for t in sorted_words],
                                            dtype=np.float32))
        qid = list(documents.keys())[-1] + 1 
        ind = defaultdict(list)
        dist = defaultdict(list)
        for i, q in enumerate(query_docs):
            # format
            text = q
            tokens = [t for t in text if t.is_alpha and not t.is_stop]
            words = Counter(t.text for t in tokens)
            orths = {t.text: t.orth for t in tokens}
            sorted_words = sorted(words)
            documents[qid] = (qid, [orths[t] for t in sorted_words],
                                np.array([words[t] for t in sorted_words],
                                            dtype=np.float32))
            # get recs
            calc = WMD(SpacyEmbeddings(), documents, vocabulary_min=1)
            for title, relevance in calc.nearest_neighbors(qid,k=num_neighbors):
                ind[i].append(title)
                dist[i].append(relevance)

    for i in range(len(queries_preprocessed)):
        recommendations["qid"].append([i for _ in range(num_neighbors)])
        recommendations["embedding"].append([emb for _ in range(num_neighbors)])
        recommendations["response_rank"].append(list(np.arange(1, num_neighbors + 1)))
        recommendations["score"].append(dist[i])
        recommendations["index"].append(ind[i])
        recommendations["response"].append(text_dataframe["answer"].iloc[ind[i]].tolist())
        recommendations["matched_q"].append(text_dataframe["question"].iloc[ind[i]].tolist())

    sm = [] # sequence match
    cs = [] # cosine similarity
    for i, correct_answer in enumerate(answer_series):
        answers = text_dataframe["answer"].iloc[ind[i]][:5]
        c_answer = answer_series[i]
        scores = [ difflib.SequenceMatcher(lambda s: not str.isalnum(s), answer, c_answer).ratio() for answer in answers]
        sm.append(torch.tensor(scores)) 
        scores = [ torch.tensor(normalized_dot_product(answer,c_a_vectors[i])) for answer in a_vectors[ind[i]]]
        cs.append(torch.stack(scores))
    
    sm = torch.stack(sm)
    torch.save(sm,'/datadrive/eval/sm_'+str(emb)+'.pkl')

    cs = torch.stack(cs)
    torch.save(cs,'/datadrive/eval/cs_'+str(emb)+'.pkl')


    ss_a_text = text_dataframe["answer"].iloc[ss_a_indices]
    ss_a_vectors = a_vectors[ss_a_indices]

    sm_all = []
    for a, b in itertools.combinations(ss_a_text, 2):
        s = difflib.SequenceMatcher(lambda s: not str.isalnum(s), a, b).ratio()
        s = torch.tensor(s)
        sm_all.append(s)

    sm_all = torch.stack(sm_all)
    torch.save(sm_all,'/datadrive/eval/sm_all_'+str(emb)+'.pkl')

    cs_all = []
    for a, b in itertools.combinations(ss_a_vectors, 2):
        s = normalized_dot_product(a, b)
        s = torch.tensor(s)
        cs_all.append(s)

    cs_all = torch.stack(cs_all)
    torch.save(cs_all,'/datadrive/eval/cs_all_'+str(emb)+'.pkl')

    if 'glove' in emb:
        wmd = []
        for i, correct_answer in enumerate(answer_series):
            wmd_b = []
            b = c_a_docs[i]
            for a in [a_docs[ind[i][j]] for j in range(5)]:
                try:
                    s = wmd_instance.compute_similarity(a,b)
                except:
                    s = float('inf')
                s = torch.tensor(s)
                wmd_b.append(s)
            wmd.append(torch.stack(wmd_b))

        wmd = torch.stack(wmd)
        torch.save(wmd,'/datadrive/eval/wmd_'+str(emb)+'.pkl')

        ss_a_docs = [a_docs[j] for j in ss_a_indices]
        wmd_all = []
        for a, b in itertools.combinations(ss_a_docs, 2):
            try:
                s = wmd_instance.compute_similarity(a,b)
            except:
                s = float('inf')
            s = torch.tensor(s)
            wmd_all.append(s)
        wmd_all = torch.stack(wmd_all)
        torch.save(wmd_all,'/datadrive/eval/wmd_all_'+str(emb)+'.pkl')


for key in recommendations.keys():
    recommendations[key] = [item for sublist in recommendations[key] for item in sublist]


torch.save(recommendations,'/datadrive/eval/recommendations.pkl')


#%%

#manual eval

recs = torch.load('/datadrive/eval/recommendations.pkl')
recs_df = pd.DataFrame(recs)

with open("/datadrive/eval/questions.pkl", "rb") as handle:
    question_series = pickle.load(handle)


with open("/datadrive/eval/answers.pkl", "rb") as handle:
    answer_series = pickle.load(handle)

#%%

eval_data = pd.DataFrame({'Q':question_series,'A':answer_series})

embeddings = ['tfidf','glove','glove_wmd','use']
from collections import defaultdict
top_1 = defaultdict(list)
for emb in embeddings:
    top_1[emb] = recs_df[(recs_df['embedding']==emb)&(recs_df['response_rank']==1)]['response']
top_1 = pd.DataFrame(top_1)

eval_data = eval_data.join(top_1)

top_1_scores = np.zeros((56,4))

# eval each qn index i
i = 0
eval_data.iloc[i].values
top_1_scores[i,:] += [0,0,0,0] # 0 or 1

# then top-5 for the ones where top-1 fails for each embedding index j for each question i in ids

top_5_scores = top_1_scores.copy()

ids = np.where(top_1_scores[:,embeddings[j].index]==0)

print(eval_data.iloc[ids[i]]['Q'])
recs_df[(recs_df['embedding']==embeddings[j])&(recs_df['qid']==ids[i])]['response'].values
top_5_scores[ids[i],embeddings[j].index] += 0 # 0 or 1



#%%

torch.save(top_1_scores,'/datadrive/eval/top_1_scores.pkl')
torch.save(top_1_scores,'/datadrive/eval/top_5_scores.pkl')
