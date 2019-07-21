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
def level1_check():
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
		for key in lev1.keys():
			print(key, lev1[key])
		print('-------------------------------------')
		# author
		for key in authors.keys():
			print(key, authors[key])
		print('-------------------------------------')
		# type
		for key in types.keys():
			print(key, types[key])
		print('-------------------------------------')
		# source
		for key in sources.keys():
			print(key, sources[key])


# level 2 check
def level2_check():
	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		lev2 = {}
		types = {}
		subtypes = {}
		cnt = 1
		for line in tqdm(f):
			obj = json.loads(line)
			contents = obj['contents']
			for li in contents:
				if type(li).__name__ == 'dict':
					for key in li.keys():
						map_cnt(lev2, key)
						if key == 'type':
							map_cnt(types, li[key])
						elif key == 'subtype':
							map_cnt(subtypes, li[key])
				# contain null field
				else:
					print('NoneType', cnt)
			cnt += 1
		print('-------------------------------------')
		# level 2 key
		for key in lev2.keys():
			print(key, lev2[key])
		print('-------------------------------------')
		# types
		for key in types.keys():
			print(key, types[key])
		print('-------------------------------------')
		# subtypes
		for key in subtypes.keys():
			print(key, subtypes[key])


# data character
def data_character():
	with open(path_mp['DataPath'] + path_mp['WashingtonPost'], 'r', encoding='utf-8') as f:
		filter_kicker = {"Opinion": 1, "Letters to the Editor": 1, "The Post's View": 1}
		filtered = []
		topics = {}
		paragraph_length = {}
		doc_length = {}
		sentence_length = {}
		cnt = 0
		for line in tqdm(f):
			obj = json.loads(line)
			contents = obj['contents']
			skip = False
			for li in contents:
				if type(li).__name__ == 'dict':
					if 'type' in li and li['type'] == 'kicker':
						# skip filter kickers
						if li['content'] in filter_kicker.keys():
							skip = True
							break
						map_cnt(topics, li['content'])
					if 'subtype' in li and li['subtype'] == 'paragraph':
						paragraph = li['content']
						# Replace <.*?> with ""
						paragraph = re.sub(r'<.*?>', '', paragraph)
						map_cnt(paragraph_length, len(paragraph))
			cnt += 1
			if skip:
				# record the filtered line idx
				filtered.append(cnt)
				continue
		# filtered
		for idx in filtered:
			print(idx)
		print('-------------------------------------')
		# topics
		topics = sorted(topics.items(), key=lambda d: d[1], reverse=True)
		for item in topics:
			print(item[0], item[1])
		print('-------------------------------------')
		# paragraph length
		for key in sorted(paragraph_length.keys()):
			print(key, paragraph_length[key])


data_character()

