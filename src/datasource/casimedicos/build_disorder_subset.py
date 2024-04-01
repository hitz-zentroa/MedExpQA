from datasource.utils import load_jsonl, save_jsonl


def run():
	languages = ['es', 'en', 'it', 'fr']
	modes = ['train', 'dev', 'test']#['train', 'dev', 'test']
	disorder_ids = load_jsonl(f"./data/casimedicos/CasiMedicosTypesDisorders.jsonl")
	for lang in languages:
		for mode in modes:
			dataset_path = f"./data/casimedicos/{mode}.{lang}.casimedicos.grounded.jsonl"
			examples = load_jsonl(dataset_path, True)
			disorder_subset = []
			for example_id, example in examples.items():
				if example_id in disorder_ids:
					disorder_subset.append(example)
			save_jsonl(f"./data/casimedicos/{mode}.{lang}.casimedicos.grounded.disorders.jsonl", disorder_subset)

def combine_dev_test():
	languages = ['es', 'en', 'it', 'fr']
	modes = ['dev', 'test']
	disorder_ids = load_jsonl(f"./data/casimedicos/CasiMedicosTypesDisorders.jsonl")
	for lang in languages:
		disorder_subset = []
		for mode in modes:
			dataset_path = f"./data/casimedicos/{mode}.{lang}.casimedicos.grounded.jsonl"
			examples = load_jsonl(dataset_path, True)
			for example_id, example in examples.items():
				if example_id in disorder_ids:
					disorder_subset.append(example)
		save_jsonl(f"./data/casimedicos/dev_test.{lang}.casimedicos.grounded.disorders.jsonl", disorder_subset)

if __name__ == '__main__':
	combine_dev_test()