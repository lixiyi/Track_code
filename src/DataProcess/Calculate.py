import getCfg as cfg
import json
import re
from tqdm import tqdm


path_mp = cfg.get_path_conf('../path.cfg')


def map_cnt(mp, key):
	if key not in mp:
		mp[key] = 1
	else:
		mp[key] += 1


# level 1 check
with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
	lev1 = {}
	authors = {}
	types = {}
	sources = {}
	for line in tqdm(f):
		obj = json.loads(line)
		for key in obj.keys():
			map_cnt(lev1, key)
			if key == 'author':
				map_cnt(authors, obj[key])
			elif key == 'type':
				map_cnt(types, obj[key])
			elif key == 'source':
				map_cnt(sources, obj[key])
	# level 1 key
	for key in lev1:
		print(key, lev1[key])
	print('-------------------------------------')
	# author
	for key in authors:
		print(key, authors[key])
	print('-------------------------------------')
	# type
	for key in types:
		print(key, types[key])
	print('-------------------------------------')
	# source
	for key in sources:
		print(key, sources[key])
		# contents = obj['contents']
		# text = ""
		# print(len(contents))
		# cnt = 0
		# for li in contents:
		# 	print(cnt, li)
		# 	cnt += 1
		# 	if type(li).__name__ == 'dict' and li['type'] == 'sanitized_html':
		# 		content = li['content']
		# 		# remove html tags, lowercase
		# 		content = re.sub(r'<.*?>', '', content)
		# 		text += content.lower()
		# obj['text'] = text
		# del obj['contents']
		# doc = json.dumps(obj)

