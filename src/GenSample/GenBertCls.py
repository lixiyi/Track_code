import sys
sys.path.append("..")


import Recall.TFIDF as tfidf
import Recall.Topics as topic
import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import random
import numpy as np
import jieba


path_mp = cfg.get_path_conf('../path.cfg')


# extract body from give Washington Post json
# args 0: json contents
# retrun: body string
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


# split max_length string from body
# args 0: string
# return: string
def split_body(args=None):
	body, max_length = args
	w_list = jieba.cut(body)
	if len(w_list) <= max_length-2:
		return body
	head_len = (max_length - 2) / 2
	tail_len = max_length - 2 - head_len
	return ' '.join(w_list[:head_len]) + ' ' + ' '.join(w_list[-tail_len:])


# generate samples for each document
# args 0: max_length for Bert
def gen_sample(args=None):
	max_length = args[0]
	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
		kicker_filterd_mp = {}		# Record line filtered by kicker, line index start from 1
		cnt = 0
		for line in tqdm(f):
			obj = json.loads(line)
			contents = obj['contents']
			title = obj['title']
			author = obj['author']
			date = obj['published_date']
			skip = False
			body = ""
			topic_name = ""
			for li in contents:
				if type(li).__name__ == 'dict':
					if 'type' in li and li['type'] == 'kicker':
						# skip filter kickers
						topic_name = li['content']
						if topic_name in filter_kicker.keys():
							skip = True
							break
					if 'subtype' in li and li['subtype'] == 'paragraph':
						paragraph = li['content'].strip()
						# Replace <.*?> with ""
						paragraph = re.sub(r'<.*?>', '', paragraph)
						body += ' ' + paragraph
			cnt += 1
			if skip:
				kicker_filterd_mp[cnt] = 1
				continue
			# Recall By tf_idf
			body = body.strip()
			res_tfidf = tfidf.recall_by_tfidf([body, '20'])

			# Recall By topics
			res_topic = topic.recall_by_topics(topic_name)

			# Combie Recall results
			res_mask = {}
			for li in res_tfidf:
				# Filter by kicker
				if int(li) in kicker_filterd_mp:
					continue
				res_mask[int(li)] = 4
			for li in res_topic:
				# Filter by kicker
				if int(li) in kicker_filterd_mp:
					continue
				if li in res_mask:
					res_mask[int(li)] = 8
				else:
					res_mask[int(li)] = 2

			# Filter
			key_list = sorted(res_mask)
			similar_doc = {}
			similar_doc[title + '#' + author + '#' + date] = 1
			doc_cache = {}		# cache document, classify by 0, 2, 4, 8
			doc_cache[0] = []
			with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f1:
				now = 0		# index for key_list
				li_cnt = 1  # index for f1
				for li in f1:
					# meet the candidate line
					if key_list[now] == li_cnt:
						doc = json.loads(li)
						doc_title = doc['title']
						doc_author = doc['author']
						doc_date = doc['published_date']
						doc_contents = doc['contents']
						now += 1
						# Filter by date
						if int(doc_date) > int(date):
							continue
						# Filter by date + title + author
						rep_key = doc_title + '#' + doc_author + '#' + doc_date
						if rep_key in similar_doc:
							continue
						else:
							similar_doc[rep_key] = 1
						# cache document
						if res_mask[li_cnt] not in doc_cache:
							doc_cache[res_mask[li_cnt]] = []
						doc_cache[res_mask[li_cnt]].append(doc)

					# random add 100 label 0 document
					elif len(doc_cache[0]) < 100 and random.random() > 0.9:
						doc = json.loads(li)
						doc_date = doc['date']
						# Filter by kicker
						if li_cnt in kicker_filterd_mp:
							continue
						# Filter by date
						if int(doc_date) > int(date):
							continue
						doc_cache[0].append(json.loads(li))
					li_cnt += 1

			# split from body
			sen1 = split_body([body, max_length])
			# Sampling and Generate examples
			with open(cfg.OUTPUT + 'Dataset_BertCls.txt', 'r', encoding='utf-8') as out:
				# label 0, 2, 4, 8
				for label in sorted(doc_cache):
					idx = random.randint(0, len(doc_cache[label])-1)
					doc = doc_cache[idx]
					doc_body = extract_body([doc['contents']])
					sen2 = split_body([doc_body, max_length])
					out.write(str(label) + '\t' + sen1 + '\t' + sen2 + '\n')



if __name__ == "__main__":
	getattr(__import__('GenBertCls'), sys.argv[1])(sys.argv[2:])


