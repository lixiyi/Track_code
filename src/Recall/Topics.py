import re

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
						map_cnt(paragraph_length, len(paragraph.split(' ')))
						doc += ' ' + paragraph
			# doc length
			doc = doc.strip()
			map_cnt(doc_length, len(doc.split(' ')))
			# sentence length
			sentences = doc.split('. ')
			for sen in sentences:
				map_cnt(sentence_length, len(sen.split(' ')))

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
