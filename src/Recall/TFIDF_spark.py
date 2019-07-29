import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import numpy as np
from pyspark import SparkContext, SparkConf


path_mp = cfg.get_path_conf('../path.cfg')


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
	w_list = set(w_list)
	res = []
	for w in w_list:
		ds = set()
		ds.add(doc_id)
		res.append((w, ds))
	return res


# create words inverted list
# No args
# outputs: words_index [length, doc line number]
# 		 : words_map (word, words_index line number)
def words_index(args = None):
	SparkContext.getOrCreate().stop()
	conf = SparkConf().setMaster("local[*]").setAppName("words_index")
	sc = SparkContext(conf=conf)
	filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
	WashingtonPost = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost'])
	WashingtonPost.flatMap(lambda line: words_index_single(line, filter_kicker)) \
		.filter(lambda w: w != ()) \
		.reduceByKey(lambda a, b: a | b) \
		.map(lambda w: str(w[0]) + ' ' + ' '.join(w[1])) \
		.saveAsTextFile(cfg.OUTPUT + 'words_index')
	sc.stop()


# tf-idf result for each document
def tfidf_index(args = None):
	# read tfidf words_mp and words_idx
	words_mp = {}
	with open(cfg.OUTPUT + 'words_index.txt', 'r', encoding='utf-8') as f:
		for line in tqdm(f):
			li = line.split(' ')
			words_mp[li[0]] = li[1:]
	SparkContext.getOrCreate().stop()
	conf = SparkConf().setMaster("local[*]").setAppName("tfidf_index")
	sc = SparkContext(conf=conf)
	filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
	WashingtonPost = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost'])
	WashingtonPost.map(lambda line: tfidf_index_single(line, filter_kicker, words_mp, 20)) \
		.saveAsTextFile(cfg.OUTPUT + 'tfidf_index')
	sc.stop()


# read words_mp and words_idx into memory first(idx start from 1)
def tfidf_index_single(line, filter_kicker, words_mp, num):
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
	num = int(num)
	# calculate term frequency for each word in the str
	tf = {}
	for w in w_list:
		if w in tf:
			tf[w] += 1
		else:
			tf[w] = 1
	# calculate idf and tf-idf for each word
	tfidf_val = {}
	for w in w_list:
		# word not in vocabulary
		if w not in words_mp:
			continue
		idf = np.log(cfg.DOCUMENT_COUNT * 1.0 / len(words_mp[w]))
		tfidf_val[w] = tf[w] * 1.0 * idf
	# sort by tf-idf, combine top inverted file line number list
	tfidf_val = sorted(tfidf_val.items(), key=lambda d: d[1], reverse=True)
	res = set()
	for i in range(min(num, len(tfidf_val))):
		w = tfidf_val[i][0]
		res = res | set(words_mp[w])
	return doc_id + ' ' + ' '.join(res)


if __name__ == "__main__":
	getattr(__import__('TFIDF_spark'), sys.argv[1])(sys.argv[2:])

