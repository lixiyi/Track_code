import getCfg as cfg
import json
import re
from tqdm import tqdm
import numpy as np


path_mp = cfg.get_path_conf('../path.cfg')


def map_cnt(mp, key):
	if key not in mp:
		mp[key] = 1
	else:
		mp[key] += 1


# data character
def data_character():
	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
		topics = {}
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
						map_cnt(topics, li['content'])
					if 'subtype' in li and li['subtype'] == 'paragraph':
						paragraph = li['content'].strip()
						# Replace <.*?> with ""
						paragraph = re.sub(r'<.*?>', '', paragraph)
						doc += ' ' + paragraph
			doc = doc.strip()
