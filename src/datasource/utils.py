import os
import json
import six


def load_jsonl(jsonl_path:str, indexed:bool=True):
	"""
	Loads the jsonl dataset into an indexed dictionary with the following structure {example_id: example}
	:param jsonl_path: path to jsonl dataset
	:param indexed:
	:return:
	"""
	d = {}
	with open(jsonl_path, "r", encoding="utf-8") as f:
		for line in f:
			line = six.ensure_text(line, "utf-8")
			example = json.loads(line)
			d[example['id']] = example
	return d if indexed else list(d.values())

def save_jsonl(out_path:str, dataset:list):
	with open(os.path.join(out_path), 'w', encoding='utf8') as outfile:
		for sample in dataset:
			json.dump(sample, outfile, ensure_ascii=False)
			outfile.write('\n')