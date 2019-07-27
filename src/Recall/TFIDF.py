import sys
sys.path.append("..")
sys.path.append("/home/trec7/lianxiaoying/bert/")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import numpy as np
import tokenization


path_mp = cfg.get_path_conf('../path.cfg')
tokenizer = tokenization.FullTokenizer(vocab_file=cfg.BERT_MODEL + 'vocab.txt', do_lower_case=False)


# create words inverted list
# No args
# outputs: words_index [length, doc line number]
# 		 : words_map (word, words_index line number)
def words_index(args = None):
	words = {}
	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
		cnt = 0
		for line in tqdm(f):
			obj = json.loads(line)
			contents = obj['contents']
			skip = False
			doc = ""
			for li in contents:
				if type(li).__name__ == 'dict':
					if 'type' in li and li['type'] == 'kicker':
						# skip filter kickers
						if li['content'] in filter_kicker.keys():
							skip = True
							break
					if 'subtype' in li and li['subtype'] == 'paragraph':
						paragraph = li['content'].strip()
						# Replace <.*?> with ""
						paragraph = re.sub(r'<.*?>', '', paragraph)
						doc += ' ' + paragraph
			cnt += 1
			if skip:
				continue
			# get inverted words for each doc
			doc = doc.strip()
			word_list = tokenizer.tokenize(doc)
			for w in word_list:
				if w not in words:
					words[w] = set()
				words[w].add(str(cnt))
	# output inverted list, first column is list length
	words_mp = {}
	with open(cfg.OUTPUT + 'words_index.txt', 'w', encoding='utf-8') as f:
		cnt = 1
		for key in sorted(words.keys()):
			words_mp[key] = cnt
			cnt += 1
			li = words[key]
			f.write(str(len(li)) + ' ' + ' '.join(li) + '\n')
	# output word to line map
	with open(cfg.OUTPUT + 'words_map.txt', 'w', encoding='utf-8') as f:
		f.write(json.dumps(words_mp))
			

# calculate tfidf for a string
# document args 1: s
# top words count args 2: num
# return: top doc line number list
def recall_by_tfidf(args = None):
	s, num = args
	num = int(num)
	# load inverted word to line map
	words_mp = {}
	with open(cfg.OUTPUT + 'words_map.txt', 'r', encoding='utf-8') as f:
		for line in f:
			words_mp = json.loads(line)
	word_list = tokenizer.tokenize(s)
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
	with open(cfg.OUTPUT + 'words_index.txt', 'r', encoding='utf-8') as f:
		cnt = 1			# line number
		now = 0			# current word index
		for line in f:
			# all the words for this document have calculated
			if now >= len(w_list):
				break
			w = w_list[now]
			# meet the right line
			if cnt == int(words_mp[w]):
				idf = np.log(cfg.DOCUMENT_COUNT * 1.0 / int(line.split(' ')[0]))
				tfidf_mp[w] = tf[w] * 1.0 * idf
				now += 1
				inv_list[w] = line.split(' ')[1:-1]
			cnt += 1
	# sort by tf-idf, combine top inverted file line number list
	tfidf_mp = sorted(tfidf_mp.items(), key=lambda d: d[1], reverse=True)
	res = []
	for i in range(min(num, len(tfidf_mp))):
		w = tfidf_mp[i][0]
		res.append(inv_list[w])
	return res


if __name__ == "__main__":
	getattr(__import__('TFIDF'), sys.argv[1])(sys.argv[2:])

