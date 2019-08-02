import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import random
import numpy as np
from pyspark import SparkContext, SparkConf
import bm25


path_mp = cfg.get_path_conf('../path.cfg')


def get_mapping(args=None):
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
	print('words_df loaded.')
	# avgdl
	avgdl = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost']) \
		.map(lambda line: bm25.calc_doc_length(line)).sum()
	avgdl = avgdl * 1.0 / 595037
	print('avgdl loaded.')
	# WashingtonPost
	WashingtonPost = sc.textFile(path_mp['DataPath'] + path_mp['WashingtonPost']) \
		.map(lambda line: bm25.return_doc(line)).collectAsMap()
	print('WashingtonPost loaded.')
	# test case: doc_id, topic_id
	case_mp = {}
	with open(path_mp['DataPath'] + path_mp['topics'], 'r', encoding='utf-8') as f:
		li = []
		for line in f:
			topic_id = re.search(r'<num>.*?</num>', line)
			if topic_id is not None:
				topic_id = topic_id.group(0)[5+9:-7]
				li.append(topic_id)
			doc_id = re.search(r'<docid>.*?</docid>', line)
			if doc_id is not None:
				doc_id = doc_id.group(0)[7:-8]
				li.append(doc_id)
			if len(li) == 2:
				case_mp[li[1]] = li[0]
				li = []
	print('test case loaded.')
	# answer: topic_id, (doc_id, rel)
	ans_mp = {}
	with open(path_mp['DataPath'] + path_mp['bqrels'], 'r', encoding='utf-8') as f:
		for line in f:
			li = line[:-1].split(' ')
			topic_id = li[0]
			doc_id = li[2]
			if topic_id not in ans_mp:
				ans_mp[topic_id] = []
			ans_mp[topic_id].append([doc_id, li[3]])
	# generate relevance map
	rel_mp = {}
	for cur_id in case_mp.keys():
		obj = WashingtonPost[cur_id]
		topic_id = case_mp[cur_id]
		# query (modify)
		query = cfg.word_cut(obj['title'])
		rel_mp[topic_id] = []
		for doc_id, rel in ans_mp[topic_id]:
			score = bm25.calc_score(WashingtonPost[doc_id], words_df, query, avgdl, True)
			rel_mp[topic_id].append([score, rel])
	with open(cfg.OUTPUT + 'rel_mp.txt', 'w', encoding='utf-8') as f:
		f.write(json.dumps(rel_mp))


# modify bm25 score in col4 to rel
def transform(args=None):
	score2rel = {}
	with open(cfg.OUTPUT + 'score2rel.txt', 'r', encoding='utf-8') as f:
		for line in f:
			score2rel = json.loads(line)
	with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/bresult.test', 'r', encoding='utf-8') as f:
		with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/bresult.test1', 'w', encoding='utf-8') as out:
			for line in f:
				li = line[:].split(' ')
				topic_id = li[0]
				li[4] = score2rel[topic_id][li[4]]
				out.write(' '.join(li))


if __name__ == "__main__":
	getattr(__import__('Score2Rel'), sys.argv[1])(sys.argv[2:])

