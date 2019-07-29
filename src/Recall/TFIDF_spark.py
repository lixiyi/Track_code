import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import numpy as np
from pyspark import SparkContext


path_mp = cfg.get_path_conf('../path.cfg')
SparkContext.getOrCreate().stop()
sc = SparkContext('local[*]', 'tfidf')


# return (word, id)
def words_index_single(line, filter_kicker):
	obj = json.loads(line)
	doc_id = obj['id']
	contents = obj['contents']
	doc = ""
	for li in contents:
		if type(li).__name__ == 'dict':
			if 'type' in li and li['type'] == 'kicker':
				# skip filter kickers
				if li['content'] in filter_kicker.keys():
					return ()
			if 'subtype' in li and li['subtype'] == 'paragraph':
				paragraph = li['content'].strip()
				# Replace <.*?> with ""
				paragraph = re.sub(r'<.*?>', '', paragraph)
				doc += ' ' + paragraph
	doc = doc.strip()
	w_list = cfg.word_cut(doc)
	res = set()
	for w in w_list:
		res.add((w, doc_id))
	return res


# create words inverted list
# No args
# outputs: words_index [length, doc line number]
# 		 : words_map (word, words_index line number)
def words_index(args = None):
	words = {}
	filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
	WashingtonPost = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost'])
	WashingtonPost.flatMap(lambda line: words_index_single(line, filter_kicker)) \
		.filter(lambda w: w != ()) \
		.reduceByKey(lambda a, b: (a[0], a[1].add(b[1]))) \
		.map(lambda w: str(w[0]) + ' ' + ' '.join(w[1])) \
		.saveAsTextFile(cfg.OUTPUT + 'words_index')


# tf-idf result for each document
def tfidf_index(args = None):
	nlp = StanfordCoreNLP(cfg.STANFORDNLP)

	# read tfidf words_mp and words_idx
	words_mp = {}
	with open(cfg.OUTPUT + 'words_map.txt', 'r', encoding='utf-8') as f:
		for line in f:
			words_mp = json.loads(line)
	words_idx = []
	words_idx.append(' ')
	with open(cfg.OUTPUT + 'words_index.txt', 'r', encoding='utf-8') as f:
		for line in tqdm(f):
			words_idx.append(line)
	print('TF-IDF idx loaded.')

	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		with open(cfg.OUTPUT + 'tfidf_index.txt', 'w', encoding='utf-8') as out:
			for line in tqdm(f):
				obj = json.loads(line)
				contents = obj['contents']
				body = ""
				for li in contents:
					if type(li).__name__ == 'dict':
						if 'subtype' in li and li['subtype'] == 'paragraph':
							paragraph = li['content'].strip()
							# Replace <.*?> with ""
							paragraph = re.sub(r'<.*?>', '', paragraph)
							body += ' ' + paragraph
				res_tfidf = cal_tfidf([body, '20', nlp, words_mp, words_idx])
				out.write(' '.join(res_tfidf) + '\n')
	nlp.close()


# read words_mp and words_idx into memory first(idx start from 1)
def cal_tfidf(args = None):
	s, num, nlp, words_mp, words_idx = args
	num = int(num)
	word_list = nlp.word_tokenize(s)
	# calculate term frequency for each word in the str
	tf = {}
	for w in word_list:
		if w in tf:
			tf[w] += 1
		else:
			tf[w] = 1
	# calculate idf and tf-idf for each word
	w_list = sorted(tf)
	tfidf_mp = {}
	inv_list = {}		# words inverted list cache
	for w in w_list:
		# word not in vocabulary
		if w not in words_mp:
			continue
		# meet the right line
		cnt = int(words_mp[w])
		line = words_idx[cnt]
		idf = np.log(cfg.DOCUMENT_COUNT * 1.0 / int(line.split(' ')[0]))
		tfidf_mp[w] = tf[w] * 1.0 * idf
		inv_list[w] = line.split(' ')[1:-1]
	# sort by tf-idf, combine top inverted file line number list
	tfidf_mp = sorted(tfidf_mp.items(), key=lambda d: d[1], reverse=True)
	res = set()
	for i in range(min(num, len(tfidf_mp))):
		w = tfidf_mp[i][0]
		res = res | set(inv_list[w])
	res = list(res)
	return res


if __name__ == "__main__":
	getattr(__import__('TFIDF_spark'), sys.argv[1])(sys.argv[2:])
	sc.stop()

