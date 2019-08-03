import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import jieba
import re
import numpy as np
from elasticsearch import Elasticsearch
from stanfordcorenlp import StanfordCoreNLP

# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
es = Elasticsearch()
nlp = StanfordCoreNLP('http://localhost', port=7000)


def test_backgound_linking():
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
	with open('/home/trec7/lianxiaoying/trec_eval.9.0/test/elastic_bresult.test', 'w', encoding='utf-8') as f1:
		for doc_id in case_mp:
			# search by docid to get the query
			dsl = {
				'query': {
					'match': {
						'id': doc_id
					}
				}
			}
			res = es.search(index='news', body=dsl)
			# print(res)
			doc = res['hits']['hits'][0]['_source']
			dt = doc['published_date']
			# make query
			# ner_filt = {'O':1, 'MONEY':1, 'NUMBER':1}
			# tmp = nlp.ner(doc['title_body'])
			# qr = []
			# for w, nn in tmp:
			# 	if nn not in ner_filt:
			# 		qr.append(w)
			# qr = ' '.join(qr)
			# query the doc
			qr = doc['title']
			dsl = {
				"size": 1000,
				"timeout": "1m",
				"query": {
					'bool': {
						'must': {
							'match': {'title_body': qr}
						},
						"must_not": {"match": {"title_author_date": doc['title_author_date']}},
						'filter': {
							"range": {"published_date": {"lte": dt}}
						}
					},
				}
			}
			res = es.search(index='news', body=dsl)
			res = res['hits']['hits']
			# output result.test file
			print(doc_id, len(res))
			cnt = 1
			for ri in res:
				out = []
				out.append(case_mp[doc_id])
				out.append('Q0')
				out.append(ri['_source']['id'])
				out.append(str(cnt))
				out.append(str(ri['_score']))
				out.append('ICTNET')
				ans = "\t".join(out) + "\n"
				f1.write(ans)
				cnt += 1


test_backgound_linking()
