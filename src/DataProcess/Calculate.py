import src.DataProcess.getCfg as cfg
import json
import re


path_mp = cfg.get_path_conf('src/path.cfg')


# check data format
with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
	lev1 = {}
	for line in tqdm(f):
		obj = json.loads(line)
		for key in obj.keys():
			if key not in lev1:
				lev1[key] = 1
			else:
				lev1[key] += 1
	for key in lev1:
		print(key, lev1[key])
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

