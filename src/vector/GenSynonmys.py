import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import jieba
import re
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors


keywords = {}
with open('keywords.txt', 'r', encoding='utf-8') as f:
	for w in f:
		w = w[:-1]
		keywords[w] = 1
print('keywords loaded.')


# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
model = KeyedVectors.load_word2vec_format('~/lianxiaoying/data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
for w in tqdm(keywords):
	w_list = model.most_similar(w, 10)

