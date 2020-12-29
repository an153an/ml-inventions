import re
import pandas as pd
import operator
from sklearn.feature_extraction.text import CountVectorizer

def get_source(df, get_task):
    s_text = df[(df.Task == get_task) & (df.Class == -1)]['Text'].iloc[0]
    return s_text

def get_answer(df, answer_filename):
    a_text, a_task = df[df.File == answer_filename][['Text', 'Task']].iloc[0]
    return a_text, a_task
    

def getNgrams(get_text,get_s_text,n):
    counter = CountVectorizer(analyzer = 'word', ngram_range = (n, n))
    ngrams_array = counter.fit_transform([get_text, get_s_text]).toarray()

    count_common_ngrams = sum(min(a, s) for a, s in zip(*ngrams_array))
    count_ngrams_a = ngrams_array[0].sum()
                                                              
    return count_common_ngrams/count_ngrams_a

def get_lcs(source_words, answer_words, lcs_matrix):
    for s, s_word in enumerate(source_words, 1):
        for a, a_word in enumerate(answer_words, 1):
            if s_word == a_word:
                lcs_matrix[s][a] = lcs_matrix[s-1][a-1] + 1
            else:
                lcs_matrix[s][a] = max(lcs_matrix[s-1][a], lcs_matrix[s][a-1])
    return lcs_matrix