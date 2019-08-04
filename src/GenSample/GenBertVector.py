import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import random
import numpy as np


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
	max_length = int(max_length)
	w_list = cfg.word_cut(body)
	if len(w_list) <= max_length-2:
		return body
	head_len = int((max_length - 2) / 2)
	tail_len = int(max_length - 2 - head_len)
	return ' '.join(w_list[:head_len]) + ' '.join(w_list[-tail_len:])


# generate samples for each document
# args 0: max_length for Bert
def gen_sample(args=None):
	max_length = args[0]
	max_length = int(max_length)
	# read all the doc, load as json, line count start from 1
	WashingtonPost = {}
	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		for line in tqdm(f):
			obj = json.loads(line)
			doc_id = obj['id']
			WashingtonPost[doc_id] = obj
	print('WashingtonPost dataset loaded.')
	with open(cfg.OUTPUT + 'Dataset_BertCls.txt', 'w', encoding='utf-8') as out:
			out.write(str(4) + '\t' + sen1 + '\t' + sen2 + '\n')


if __name__ == "__main__":
	getattr(__import__('GenBertVector'), sys.argv[1])(sys.argv[2:])


