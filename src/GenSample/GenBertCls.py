import sys
sys.path.append("..")


import Recall.TFIDF as tfidf
import Recall.Topics as topic
import DataProcess.getCfg as cfg
import json
import re
from tqdm import tqdm
import numpy as np
import jieba


path_mp = cfg.get_path_conf('../path.cfg')


def genSample(args = None):
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
			res = tfidf.get_tfidf([doc, '20'])
			print(doc)
			print(res)

if __name__ == "__main__":
	getattr(__import__('GenBertCls'), sys.argv[1])(sys.argv[2:])


