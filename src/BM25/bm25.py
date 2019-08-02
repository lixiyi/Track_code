import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import random
import numpy as np
from pyspark import SparkContext, SparkConf


path_mp = cfg.get_path_conf('../path.cfg')


def extract_body(args = None):
	contents = args[0]
	body = ''
	for p in contents:
		if type(p).__name__ == 'dict':
			if 'subtype' in p and p['subtype'] == 'paragraph':
				paragraph = p['content'].strip()
				# Replace <.*?> with ""
				paragraph = re.sub(r'<.*?>', '', paragraph)
				body += ' ' + paragraph
	return body


def filter_doc(doc, date, similar_doc):
	doc_title = doc['title']
	doc_author = doc['author']
	doc_date = doc['published_date']
	# Filter by date
	if doc_date is not None and date is not None and int(doc_date) > int(date):
		return False
	# Filter by date + title + author
	rep_key = ''
	if doc_title is not None:
		rep_key += doc_title
	if doc_author is not None:
		rep_key += '#' + doc_author
	if doc_date is not None:
		rep_key += '#' + str(doc_date)
	if rep_key in similar_doc:
		return False
	similar_doc[rep_key] = 1
	return True


def calc_doc_length(line):
	obj = json.loads(line)
	body = extract_body(obj['contents'])
	w_list = cfg.word_cut(body)
	return (1, len(w_list))


def calc_score(line, words_df, query, avgdl):
	k1 = 1.5
	b = 0.75
	obj = json.loads(line)
	body = extract_body(obj['contents'])
	doc_id = obj['id']
	w_list = cfg.word_cut(body)
	# calc tf for the doc
	tf = {}
	for w in w_list:
		if w in tf:
			tf[w] += 1
		else:
			tf[w] = 1
	# calc bm25 for the doc
	score = 0.0
	for w in query:
		tfi = 0
		if w in tf:
			tfi = tf[w]
		dfi = 1e-7
		if w in words_df.value:
			dfi = words_df.value[w]
		dl = len(w_list)
		N = cfg.DOCUMENT_COUNT
		score += np.log(N / dfi) * ((k1 + 1) * tfi) / (k1 * ((1 - b) + b * dl / avgdl) + tfi)
	return (score, doc_id)


# words_df: document frequency for each word
# WashingtonPost: corpus
def bm25(query):
	SparkContext.getOrCreate().stop()
	conf = SparkConf().setMaster("local[*]").setAppName("bm25") \
		.set("spark.executor.memory", "10g") \
		.set("spark.driver.maxResultSize", "10g") \
		.set("spark.cores.max", 10) \
		.set("spark.executor.cores", 10) \
		.set("spark.default.parallelism", 20)
	sc = SparkContext(conf=conf)
	# words df
	words_df = sc.textFile(cfg.OUTPUT + 'words_index.txt') \
		.filter(lambda line: line != '') \
		.map(lambda line: (line.split(' ')[0], len(line.split(' ')[1:]))) \
		.collectAsMap()
	words_df = sc.broadcast(words_df)
	# avgdl
	avgdl = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost']) \
		.map(lambda line: calc_doc_length(line))\
		.reduceByKey(lambda a, b: a[1] + b[1]).collect()
	# avgdl = avgdl * 1.0 / 595037
	print(type(avgdl), avgdl)
	# res = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost']) \
	# 	.repartition(4000) \
	# 	.map(lambda line: calc_score(line, words_df, query, avgdl))\
	# 	.sortByKey().collect()
	# for item in res[:1000]:
	# 	print(item[0], item[1])


if __name__ == "__main__":
	getattr(__import__('bm25'), sys.argv[1])(sys.argv[2:])

