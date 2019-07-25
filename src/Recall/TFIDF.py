import sys
sys.path.append("..")


import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import numpy as np
import jieba


path_mp = cfg.get_path_conf('../path.cfg')


# create words inverted list
# No args
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
			word_list = jieba.cut_for_search(doc)
			for w in word_list:
				if w not in words:
					words[w] = []
				words[w].append(str(cnt))
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
		f.write(json.dump(words_mp))
			

# calculate tfidf for a string
# document args 1: s
# top words count args 2: num
# return: top words
def get_tfidf(args = None):
	s, num = args
	num = int(num)
	# load inverted word list
	words_mp = {}
	with open(cfg.OUTPUT + 'words_map.txt', 'r', encoding='utf-8') as f:
		for line in f:
			words_mp = json.load(line)
	word_list = jieba.cut_for_search(s)
	# calculate term frequency for each word in the str
	tf = {}
	for w in word_list:
		if w in tf:
			tf[w] += 1
		else:
			tf[w] = 1
	# calculate idf for each word
	idf = {}
	with open(cfg.OUTPUT + 'words_index.txt', 'r', encoding='utf-8') as f:
		for w in sorted(tf.keys()):
			li = words_mp[w]
			
	# calculate tf-idf for each word
	tfidf_mp = {}
	for w in tf.keys():
		idf = np.log(cfg.DOCUMENT_COUNT * 1.0 / len(words[w]))
		tfidf = tf[w] * 1.0 * idf
		tfidf_mp[w] = tfidf
	# sort by tf-idf
	tfidf_mp = sorted(tfidf_mp.items(), key=lambda d: d[1], reverse=True)
	res = []
	for i in range(num):
		res.append(tfidf_mp[i][0])
	return res


if __name__ == "__main__":
	getattr(__import__('TFIDF'), sys.argv[1])(sys.argv[2:])


