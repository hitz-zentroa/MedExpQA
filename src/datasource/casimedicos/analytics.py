import json
import os
from collections import defaultdict

import six
from transformers import AutoTokenizer

from datasource.casimedicos.dataset import STRINGS_MULTI_LANG, PROMPT_02_MEDRAG, build_prompt
from datasource.utils import load_jsonl, save_jsonl
from tqdm import tqdm
import numpy as np

from tools.analytics_utils import plot_size_dist, multiple_dist


def get_other_explanations(example):
	full_answer = example["full_answer"]
	for ans_range in  example["explanations"][str(example["correct_option"])]["char_ranges"]:
		full_answer = ''.join([full_answer[:ans_range[0]], full_answer[ans_range[1]:]])
	return full_answer

def empty_other_explanations():
	"""
	Calculates how many full_explanations turn into empty strings after correct option removal
	:return:
	"""
	for lang in ["en", "es", "it", "fr"]:
		for mode in ["train", "dev", "test"]:
			total = 0
			jsonl_path = f"data/casimedicos/{mode}.{lang}.casimedicos.grounded.jsonl"
			dataset = load_jsonl(jsonl_path, indexed=False)
			for example in dataset:
				other_explain = get_other_explanations(example)
				total += 1 if other_explain == '' and other_explain != example["full_answer"] else 0
			print(f"{lang} {mode}: {total}/{len(dataset)} ({(total/len(dataset))*100:.1f}%)")

def correct_index_dist():
	for lang in ["en"]:
		for mode in ["train", "dev", "test"]:
			total = 0
			jsonl_path = f"data/casimedicos/{mode}.{lang}.casimedicos.grounded.jsonl"
			dataset = load_jsonl(jsonl_path, indexed=False)
			corrects = []
			for example in dataset:
				corrects.append(example["correct_option"])
			for opt in [1, 2, 3, 4, 5]:
				print(f"{mode} {opt}: {corrects.count(opt)/len(corrects)*100:.1f}")



def get_dataset_tokens(lang):
	"""
	Stats for infoboxes
	:return:
	"""
	token_lengths = {"full_question":[], "options":[], "correct_exp":[], "full_answer":[]}
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
	for mode in ["train", "dev", "test"]:
		jsonl_path = f"data/casimedicos/{mode}.{lang}.casimedicos.grounded.jsonl"
		dataset = load_jsonl(jsonl_path, indexed=False)
		for example in dataset:
			token_lengths["full_question"].append(len(tokenizer(example["full_question"])[0]))
			token_lengths["options"].append(len(tokenizer(''.join([option for option in example["options"].values() if isinstance(option, str)]))[0]))
			token_lengths["correct_exp"].append(len(tokenizer(example["explanations"][str(example["correct_option"])]["text"])[0]))
			token_lengths["full_answer"].append(len(tokenizer(example["full_answer"])[0]))
	print(f"Lang: {lang}")
	for key, value in token_lengths.items():
		print(f"{key} avg: {np.average(value):.1f}, min: {np.min(value)},max: {np.max(value)}, sd: {np.std(value):.1f}")

def get_dataset_doc_stats(dataset_path, lang, max_length):
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
	dataset = load_jsonl(dataset_path, indexed=False)
	num_docs = []
	doc_soruces = []
	t_lengths = []
	for example in dataset:
		input_text, answer_text, full_text, stats = build_prompt(example, tokenizer, PROMPT_02_MEDRAG, ["rag_cc"], [lang], max_length, True)
		num_docs.append(stats["num_docs_in"])
		for doc_id in stats["doc_sources"]:
			if "pubmed" in doc_id:
				source = "pubmed"
			elif "wiki" in  doc_id:
				source = "wiki"
			elif "article" in  doc_id:
				source = "statpearls"
			else:
				source = "textbooks"
			doc_soruces.append(source)
		t_lengths.append(len(tokenizer(text=full_text, max_length=max_length, truncation=True, padding=False, return_tensors=None, add_special_tokens=True)["input_ids"]))

	return num_docs, doc_soruces, t_lengths


def analyze_docs_in_context_window(lang, max_length):
	main_path = "./data/casimedicos/"
	file_names = [f"{mode}.{lang}.casimedicos.rag.jsonl" for mode in ["train", "dev", "test"]]
	all_docs = []
	all_sources = []
	token_lengths = []
	for file_name in tqdm(file_names):
		path = os.path.join(main_path, file_name)
		num_docs, sources, t_lengths = get_dataset_doc_stats(path, lang, max_length)
		all_docs.extend(num_docs)
		all_sources.extend(sources)
		token_lengths.extend(t_lengths)

	# plot_size_dist([v for v in token_lengths if v < 1000])
	#plot_size_dist(all_docs, bins=list(range(1, 33)))

	print(f"PubMed: {len([x for x in all_sources if x == 'pubmed'])/len(all_sources)*100:.2f}")
	print(f"Wikipedia: {len([x for x in all_sources if x == 'wiki'])/len(all_sources)*100:.2f}")
	print(f"Statpearls: {len([x for x in all_sources if x == 'statpearls'])/len(all_sources)*100:.2f}")
	print(f"Textbooks: {len([x for x in all_sources if x == 'textbooks'])/len(all_sources)*100:.2f}")
	#plot_size_dist(token_lengths, title="Context lengths")
	return all_docs


if __name__ == '__main__':
	#empty_other_explanations()
	#correct_index_dist()
	for lang in ["en", "es", "it", "fr"]:
		get_dataset_tokens(lang)
	#d = {"docs":[], "length":[]}
	#for length in [2048, 4096, 8000]:
	#	docs = analyze_docs_in_context_window("en", length)
	#	d["docs"].extend(docs)
	#	d["length"].extend([str(length)]*len(docs))
	#multiple_dist(d)