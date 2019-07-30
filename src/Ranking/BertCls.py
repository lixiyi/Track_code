class BertClsProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		lines = self._read_tsv(data_dir)
		examples = []
		for (i, line) in enumerate(lines):
			guid = "train-%d" % (i)
			text_a = tokenization.convert_to_unicode(line[0])
			text_b = tokenization.convert_to_unicode(line[1])
			label = tokenization.convert_to_unicode(line[2])
			if label == tokenization.convert_to_unicode("contradictory"):
				label = tokenization.convert_to_unicode("contradiction")
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples

