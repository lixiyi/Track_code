class BertClsProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		file_path = os.path.join(data_dir, 'train.csv')
		with open(file_path, 'r') as f:
			reader = f.readlines()
		examples = []
		for index, line in enumerate(reader):
			guid = 'train-%d'%index
			split_line = line.strip().split('\t')
			text_a = tokenization.convert_to_unicode(split_line[1])
			text_b = tokenization.convert_to_unicode(split_line[2])
			label = split_line[0]
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


	def get_labels(self):
		return ['0', '2', '4', '8', '16']

