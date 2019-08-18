import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import jieba
import re
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors


# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')


keywords = {}
with open('keywords.txt', 'r', encoding='utf-8') as f:
	for w in f:
		w = w[:-1]
		keywords[w] = 1
print('keywords loaded.')


model = KeyedVectors.load_word2vec_format('~/lianxiaoying/data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
print('model loaded.')


with open('synonyms.txt', 'w', encoding='utf-8') as out:
	for w in tqdm(keywords):
		if w in model.vocab:
			w_list = model.most_similar(positive=w, topn=10)
			res = w_list[0][0]
			for wi, sim in w_list[1:]:
				res += ', ' + wi
			out.write(res + '\n')

