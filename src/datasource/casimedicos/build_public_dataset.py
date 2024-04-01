import os

from datasource.utils import load_jsonl, save_jsonl


def run(split, lang, include_rag:bool = False):
	dataset_dir = f"data/casimedicos/"
	data_clean_path = os.path.join(dataset_dir, f"{split}.{lang}.casimedicos.clean.jsonl")
	og_data_path = os.path.join(dataset_dir, f"{lang}.casimedicos.jsonl")
	data_rag_path = os.path.join(dataset_dir, f"{split}.{lang}.casimedicos.rag.jsonl")
	out_path = os.path.join(dataset_dir, "public", f"{split}.{lang}.casimedicos{'.rag' if include_rag else ''}.jsonl")
	dataset_clean = load_jsonl(data_clean_path, False)
	og_dataset = load_jsonl(og_data_path, True)
	if include_rag:
		dataset_rag = load_jsonl(data_rag_path, True)
	final_dataset = []
	for example_clean in dataset_clean:
		new_example = {}
		if example_clean["full_question"] != og_dataset[example_clean["id"]]["full_question"]:
			print(f"DIFF {lang}: {example_clean['full_question']}")
		for key, option in example_clean["options"].items():
			if isinstance(option, str) and option != og_dataset[example_clean["id"]]["options"][key]:
				print(f"DIFF {lang}: {example_clean['full_question']}")
		new_example["correct_option"] = example_clean["correct_option"]
		new_example["explanations"] = example_clean["explanations"]
		new_example["full_answer"] = example_clean["full_answer"]
		new_example["full_answer_no_ref"] = example_clean["full_answer_no_ref"]
		new_example["full_question"] = example_clean["full_question"]
		new_example["id"] = example_clean["id"]
		new_example["lang"] = example_clean["lang"]
		new_example["options"] = example_clean["options"]
		new_example["question_id_specific"] = example_clean["question_id_specific"]
		new_example["type"] = example_clean["type"]
		new_example["year"] = example_clean["year"]

		if include_rag:
			new_example["rag"] = dataset_rag[example_clean["id"]]["rag"]
		final_dataset.append(new_example)
	save_jsonl(out_path, final_dataset)


if __name__ == '__main__':
	for mode in ["train", "dev", "test"]:
		for language in ["en", "es", "it", "fr"]:
			run(mode, language, include_rag=True)