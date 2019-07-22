import sys
import getCfg as cfg
import json
import re
from tqdm import tqdm
import numpy as np


sys.path('../DataProcess')
path_mp = cfg.get_path_conf('../path.cfg')


def topics_index():
	topics = {}
	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		cnt = 1
		for line in tqdm(f):
			obj = json.loads(line)
			contents = obj['contents']
			for li in contents:
				if type(li).__name__ == 'dict':
					if 'type' in li and li['type'] == 'kicker':
						key = li['content']
						if key in topics:
							topics[key].append(cnt)
						else:
							topics[key] = []
							topics[key].append(cnt)
			cnt += 1
	with open('topics_index.txt', 'w', encoding='utf-8') as f:
		f.write(json.dumps(topics))


topics_index()
