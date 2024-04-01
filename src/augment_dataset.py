from typing import Dict
import copy

from datasource.casimedicos.dataset import GROUNDING_ENTITY_TYPES

from datasource.utils import save_jsonl, load_jsonl
from rag.medraglib.utils import RetrievalSystem


def get_rag_question(example, raq_query_type:str):
	if raq_query_type == "clinical_case_options":
		full_q = f"{example['full_question']}"
		full_q += ' '.join([f'{i} {option.strip()}' for i, option in example["options"].items() if isinstance(option, str)])
		return full_q
	elif raq_query_type == "clinical_entities":
		# "Body parts: bla, bla. Diseases: bla, bla."
		return '. '.join([f"{GROUNDING_ENTITY_TYPES[entity_type][example['lang']]}: {', '.join(entities)}" for entity_type, entities in example['clinical_entities'].items() if len(entities) > 0])

def apply_rag(dataset_paths:Dict, k=32, rrf_k=100, force_run:bool=False):
	rag_queries = ["clinical_case_options"] #["clinical_entities", "clinical_case_options"]
	rag_corpora = ["MedCorp"] # ["PubMed", "MedCorp", "StatPearls", "Textbooks", "Wikipedia"]
	retrievers = ["BM25", "RRF-2"] # ["BM25", "Contriever", "SPECTER", "MedCPT", "RRF-2", "RRF-4"]
	print("Loading casimedicos dataset")
	datasets = copy.deepcopy(dataset_paths)  # {'es': {"train": {"id":example}, "test": {"id":example}}}
	for lang, dataset_block in dataset_paths.items():
		for mode, dataset_path in dataset_block.items():
			datasets[lang][mode] = load_jsonl(dataset_path)

	retrieval_system = RetrievalSystem("RRF-2", "MedCorp", "./data/rag/")
	for lang in ["en", "es", "it", "fr"]:
		dataset_block = datasets[lang]
		for mode in ["dev", "test", "train"]:
			dataset = dataset_block[mode]
			out_path = f"./data/casimedicos/{mode}.{lang}.casimedicos.rag.jsonl"
			#if not force_run and os.path.isfile(out_path):
			#	continue
			print(f"Generating {out_path}")
			for i, (example_id, example) in enumerate(dataset.items()):
				print(f"({lang}) {mode} {i}/{len(dataset)}")
				example["rag"] = {}
				for rag_query in rag_queries:
					example["rag"][rag_query] = {}
					for corpus_name in rag_corpora:
						example["rag"][rag_query][corpus_name] = {}
						for retriever_name in retrievers:
							question = get_rag_question(example, rag_query)
							retrieved_snippets, scores = retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
							#contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
							doc_list = [{"id":snippet["id"], "title":snippet["title"], "score":scores[idx], "content":snippet["content"]} for idx, snippet in enumerate(retrieved_snippets)]
							example["rag"][rag_query][corpus_name][retriever_name] = doc_list
			save_jsonl(out_path, list(dataset.values()))


def run():
	dataset_paths = {}
	for language in ['es', 'en', 'it', 'fr']:
		dataset_paths[language] = {}
		for split in ["dev", "train", "test"]:
			dataset_paths[language][split] = f"./data/casimedicos/{split}.{language}.casimedicos.jsonl"
	# 4. Add RAG snippets
	print("Applying RAG")
	apply_rag(dataset_paths, k=32)


if __name__ == '__main__':
	run()