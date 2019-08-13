import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import os
import jieba
import re
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import KeyedVectors

# get file path conf
path_mp = cfg.get_path_conf('../path.cfg')
es = Elasticsearch(port=7200)
nlp = StanfordCoreNLP('http://localhost', port=7000)


def test_backgound_linking():
	# load glove
	glove_model = KeyedVectors.load_word2vec_format('/home/trec7/lianxiaoying/data/glove.840B.300d.word2vec.txt', binary=False)
	print('glove words vector loaded.')
	# read words_mp
	idf = {}
	with open(cfg.OUTPUT + 'words_index.txt', 'r', encoding='utf-8') as f:
		for line in tqdm(f):
			li = line[:-1].split(' ')
			idf[li[0].lower()] = len(li[1:])
	print('words idx loaded.')
	# stop words
	stop_words = {}
	with open('stopwords.txt', 'r', encoding='utf-8') as f:
		for w in f:
			w = w[:-1]
			stop_words[w] = 1
	print('stop words loaded.')
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
			# ner_filt = {'O': 1, 'MONEY': 1, 'NUMBER': 1}
			# ner_need = {
			# 	'PERSON': 1, 'LOCATION': 1, 'STATE_OR_PROVINCE': 1,
			# 	'ORGANIZATION': 1, 'CITY': 1
			# }
			# tmp1 = nlp.ner(doc['title_body'])
			# key_word = []
			# for w, nn in tmp1:
			# 	if nn in ner_need:
			# 		key_word.append(w)
			# key_word = ' '.join(key_word)
			# query the doc
			tmp1 = cfg.word_cut(doc['title_body'])
			tmp = []
			tf = {}
			for w in tmp1:
				if w not in stop_words:
					tmp.append(w)
					if w in tf:
						tf[w] += 1
					else:
						tf[w] = 1
			qr = ' '
			if len(tmp) > 768:
				qr += ' '.join(tmp[:512]) + ' ' + ' '.join(tmp[-256:])
				tmp1 = tmp[:512] + tmp[-256:]
			else:
				qr += ' '.join(tmp)
				tmp1 = tmp
			tmp = []
			for w in tmp1:
				if w in glove_model:
					sw = glove_model.most_similar(w)[0][0]
					tmp.append(sw)
			qr1 = ' '.join(tmp)
			dsl = {
				"size": 1000,
				"timeout": "1m",
				"query": {
					'bool': {
						 'must': {
						 	'match': {
								'title_body': {
									'query':qr,
									'boost':1
								}
							}
						 },
						'should': [
							 {
							 	'match': {
							 		'title_body': {
							 			'query': doc['title'],
							 			"boost": 3
							 		}
							 	}
							 },
							 {
							 	'match': {
							 		'title_body': {
							 			'query': qr1,
							 			"boost": 1
							 		}
							 	}
							 },
						],
						"must_not": {"match": {"title_author_date": doc['title_author_date']}},
						'filter': {
							"range": {"published_date": {"lte": dt}}
						}
					},
				}
			}
			# add words weight by tfidf
			#tfidf = {}
			#for w in tmp:
			#	if w not in idf:
			#		idf[w] = 1
			#	tfidf[w] = tf[w] * np.log(cfg.DOCUMENT_COUNT * 1.0 / idf[w])
			#tfidf = sorted(tfidf.items(), key=lambda d: d[1], reverse=True)
			#ed = min(20, len(tfidf))
			#maxsc = tfidf[0][1]
			#minsc = tfidf[-1][1]
			#for w, sc in tfidf[:ed]:
			#	if w in glove_model:
			#		sw = glove_model.most_similar(w)[0][0]
			#		mpi = {
			#			'match': {
			#				'title_body': {
			#					'query': sw,
			#					"boost": 1#4 + (sc - minsc)*1.0/(maxsc - minsc)
			#				}
			#			}
			#		}
			#		dsl['query']['bool']['should'].append(mpi)
			# search
			res = es.search(index='news', body=dsl, request_timeout=30)
			res = res['hits']['hits']
			# output result.test file
			print(doc_id, len(res))
			cnt = 1
			rep_mp = {}
			rep_mp[doc['title_author_date']] = 1
			for ri in res:
				rep_key = ri['_source']['title_author_date']
				if rep_key in rep_mp:
					continue
				else:
					rep_mp[rep_key] = 1
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
