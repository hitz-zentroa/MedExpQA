import glob
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from functools import partial
from itertools import chain
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Optional, Sized, Union

import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.utils.data import Dataset

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datasource.grounding.preprocessing import linearize_infobox
from model.config import GOAL_ANS_IDX, GOAL_ARG_GEN, PROMPT_01_CLEAN, PROMPT_02_MEDRAG

GROUNDING_ENTITY_TYPES = {
	"body-part": {"es":"Partes del cuerpo", "en":"Body parts", "it":"Parti del corpo", "fr":"Parties du corps"},
	"disabilities": {"es":"Discapacidades", "en":"Disabilities", "it":"Disabilità", "fr":"Handicapés"},
	"disease": {"es":"Enfermedades", "en":"Diseases", "it":"Malattie", "fr":"Maladies"},
	"finding": {"es":"Resultados", "en":"Findings", "it":"Reperti", "fr":"Résultats"},
	"medication": {"es":"Medicaciones", "en":"Medication", "it":"Farmaco", "fr":"Médicament"},
	"procedure": {"es":"Procedimientos", "en":"Procedure", "it":"Procedura", "fr":"Procédure"},
	#"proteins": {"es":"Proteínas", "en":"Proteins", "it":"Proteine", "fr":"Protéines"},
	"tumours": {"es":"Tumores", "en":"Tumours", "it":"Tumori", "fr":"Tumeurs"}
}
GROUNDING_SPECIAL_ENTITIES = {
	"explanation_full":"explanation_full",
	"explanation_correct":"explanation_correct",
	"explanation_other":"explanation_other",
	"explanation_clean":"explanation_clean",
	"rag_cc":"rag_cc",
	"rag_ce":"rag_ce",
}

STRINGS_MULTI_LANG = {
	"option": {"es":"Opción","en":"Option","it":"Opzione","fr":"Option"},
	"options": {"es":"Opciones","en":"Options","it":"Opzioni","fr":"Choix"},
	"correct": {"es":"correcta","en":"correct","it":"corretta","fr":"correcte"},
	"incorrect": {"es":"incorrecta","en":"incorrect","it":"errata","fr":"incorrecte"},
	"additional_info": {"es":"Información adicional","en":"Additional info","it":"informazioni addizionali","fr":"information additionnelle"},
	"explanation": {"es":"Explicación","en":"Explanation","it":"Spiegazione","fr":"Explication"},
    PROMPT_02_MEDRAG: {"es":"", "en":"", "it":"", "fr":""},
    PROMPT_01_CLEAN: {"es": " La opción correcta es: ", "en": " The correct answer is: ", "it": " La risposta corretta è: ", "fr": " La bonne réponse est: "},
}

PROMPT_TEMPLATES = {
	PROMPT_02_MEDRAG: {
		"en": ("You are a helpful medical expert, and your task is to answer a multi-choice medical question using"
				" the relevant documents. Please choose the answer from the provided options. Your responses will "
				"be used for research purposes only, so please have a definite answer.\n"
				"Here are the relevant documents:\n"
				"{grounding}\n"
				"Here is the question:\n"
				"{question}\n"
				"Here are the potential choices:\n"
				"{options}\n"
				"The correct answer is: {answer}"),
		"es": ("Eres un experto médico y tu tarea consiste en responder a una pregunta médica de test utilizando tu "
			  "conocimiento y los siguientes documentos relevantes. Por favor, elige la respuesta entre las opciones proporcionadas. "
			  "Tus respuestas se utilizarán únicamente con fines de investigación, así que te rogamos que proporciones una "
			  "respuesta definitiva.\n"
			  "Estos son los documentos relevantes:\n"
			  "{grounding}\n"
			  "Aquí está la pregunta:\n"
			  "{question}\n"
			  "Aquí están las posibles opciones:\n"
			  "{options}\n"
			  "La opción correcta es: {answer}"),
		"it": ("Sei un medico esperto e il tuo compito consiste nel rispondere a una domanda di test medico utilizzando le "
			  "tue conoscenze e i documenti successivi rilevanti. Per favore, scegli la risposta tra le opzioni fornite. Le "
			  "tue risposte verranno utilizzate esclusivamente con fini di indagine, quindi ti chiediamo di fornirti una "
			  "risposta definitiva.\n"
			  "Questi sono i documenti rilevanti:\n"
			  "{grounding}\n"
			  "Ecco la domanda:\n"
			  "{question}\n"
			  "Ecco le opzioni possibili:\n"
			  "{options}\n"
			  "L'opzione corretta è: {answer}"),
		"fr": ("Vous êtes un expert en médecine et votre tâche consiste à répondre à une question d’examen médical en "
			  "utilisant vos connaissances et les documents suivants. Veuillez choisir la réponse parmi les options "
			  "proposées. Vos réponses seront utilisées uniquement à des fins de recherche, veuillez donc fournir une "
			  "réponse claire.\n"
			   "Voici les documents pertinents:\n"
			  "{grounding}\n"
			  "Voici la question:\n"
			  "{question}\n"
			  "Voici les options possibles:\n"
			  "{options}\n"
			  "La bonne option est: {answer}"),
	}
}

MASKED = "[HIDDEN]"

def build_grounding(priority_text:str, grounding_entities: List[str], groundings, grounding_langs:List[str],
					tokenizer: PreTrainedTokenizerBase, max_length:int):
	"""
	Builds the grounding up to max_length taking into consideration the composition of the grounding entities (this is,
	it cuts entire entities, not just tokens)
	:param priority_text: text reserved for other tokens of the prompt with a higher trunkation priority than grounding
	:param grounding_entities:
	:param groundings:
	:param grounding_langs:
	:param tokenizer:
	:param max_length:
	:return:
	"""

	grounding_lines = []
	taken_len = len(tokenizer(
				text=priority_text,
				max_length=max_length,
				truncation=True,
				padding=False,
				return_tensors=None,
				add_special_tokens=True,
			)["input_ids"])
	length_left = max_length - taken_len
	if length_left <= 0:
		return "", 0
	non_infobox_in_grounding = False
	for grounding_lang in grounding_langs:
		for entity_type in grounding_entities:
			if grounding_lang in groundings:
				for entity in groundings[grounding_lang][entity_type]:
					if entity_type in GROUNDING_ENTITY_TYPES.keys():
						# Infobox linearization
						grounding_lines.append(" " + linearize_infobox(entity["entity"], entity["data"]))
					else:
						# Plain text grounding line
						grounding_lines.append(" " + entity)
						non_infobox_in_grounding = True
	if len(grounding_lines) == 0:
		return "", 0
	# Remove duplicates and maintain order
	grounding_lines = list(dict.fromkeys(grounding_lines))
	tok_grounding_lines = tokenizer(
		text=grounding_lines,
		max_length=length_left if non_infobox_in_grounding else max_length,
		truncation=True,
		padding=False,
		return_tensors=None,
		add_special_tokens=True,
	)["input_ids"]
	assert len(grounding_lines) == len(tok_grounding_lines)
	grounding = ""
	num_docs_in = 0
	for g_line, tok_line in zip(grounding_lines, tok_grounding_lines):
		if len(tok_line) <= length_left:
			#grounding += " " if len(grounding) > 0 else ""
			#grounding += g_line # WATCH OUT!!! THIS LINE IS NOT TRUNCATED
			# We need to detokenize the truncated line, otherwise we might be appending the original non-truncated line
			grounding += tokenizer.decode(tok_line, skip_special_tokens=True)
			length_left -= len(tok_line)
			num_docs_in += 1
		else:
			break
	return grounding, num_docs_in


def build_prompt(example:dict,
				 tokenizer: PreTrainedTokenizerBase,
				 prompt_style: str,
				 grounding_entities: List[str],
				 grounding_langs: List[str],
				 max_length: int,
				 calc_stats:bool=False,
				 max_rag_docs:int = None):

	input_text = ""
	answer_text = ""
	full_text = ""
	num_docs_in = 0
	stats = None
	full_question = example["full_question"].strip() if "full_question" in example else ""
	correct_option = int(example["correct_option"]) if "correct_option" in example else ""
	options = example["options"] if "options" in example else ""
	qtype = example["type"].strip() if "type" in example else ""
	full_answer = example["full_answer"] if "full_answer" in example else ""
	correct_explanation = example["explanations"][str(correct_option)]["text"] if "explanations" in example else ""
	other_explanation = get_other_explanations(example) if "full_answer" in example else ""
	clean_explanation = example["full_answer_no_ref"] if "full_answer_no_ref" in example else ""
	rag_cc = [doc["content"] for doc in example["rag"]["clinical_case_options"]["MedCorp"]["RRF-2"]] if "rag" in example else []
	groundings = example["grounding"] if "grounding" in example else {"es":{},"en":{},"fr":{},"it":{}}
	for lang, grounding in groundings.items():
		grounding[GROUNDING_SPECIAL_ENTITIES["explanation_full"]] = [full_answer]
		grounding[GROUNDING_SPECIAL_ENTITIES["explanation_correct"]] = [correct_explanation]
		grounding[GROUNDING_SPECIAL_ENTITIES["explanation_other"]] = [other_explanation]
		grounding[GROUNDING_SPECIAL_ENTITIES["explanation_clean"]] = [clean_explanation]
		grounding[GROUNDING_SPECIAL_ENTITIES["rag_cc"]] = rag_cc if max_rag_docs is None else rag_cc[:max_rag_docs]
	lang = example["lang"] if "lang" in example else 'es'
	grounding_entities = grounding_entities if grounding_entities is not None else []
	if GROUNDING_SPECIAL_ENTITIES['explanation_clean'] in grounding_entities and "full_answer_no_ref" not in example:
		logging.warning(f"'{GROUNDING_SPECIAL_ENTITIES['explanation_clean']}' is one of the grounding sources but this dataset doesn't contain 'full_answer_no_ref'. Are you using the right dataset?")
	grounding_langs = grounding_langs if grounding_langs is not None else [lang] # Default grounding lang is example_lang

	if prompt_style == PROMPT_01_CLEAN:
		# Legacy way, this should be refactored and merged with PROMPT_02_MEDRAG implementation
		text_answer = f"{correct_option} {options[str(example['correct_option'])].strip()}"
		text = f"{qtype}: {full_question} "
		text += ' '.join([f'{STRINGS_MULTI_LANG["option"][lang]}: {i} {option.strip()}' for i, option in options.items() if
						  isinstance(option, str)])
		taken_text = text + "//" + STRINGS_MULTI_LANG[prompt_style][lang] + text_answer
		if len(grounding_entities) > 0:
			grounding_text, num_docs_in = build_grounding(taken_text, grounding_entities, groundings, grounding_langs, tokenizer, max_length)
			text += grounding_text
		input_text = text
		answer_text = text_answer
		full_text = input_text + answer_text
	elif prompt_style == PROMPT_02_MEDRAG:
		template = PROMPT_TEMPLATES[PROMPT_02_MEDRAG][lang]
		question_text = f"{qtype}: {full_question}"
		options_text = "\n".join([f'{i}. {option.strip()}' for i, option in options.items() if isinstance(option, str)])
		answer_text = f"{correct_option}. {options[str(example['correct_option'])].strip()}"
		priority_text = template.format(grounding="", question=question_text, options=options_text, answer=f"{correct_option}.")
		grounding_text, num_docs_in = build_grounding(priority_text, grounding_entities, groundings, grounding_langs, tokenizer, max_length)
		input_text = template.format(grounding=grounding_text, question=question_text, options=options_text, answer="")
		full_text = template.format(grounding=grounding_text, question=question_text, options=options_text, answer=answer_text)
	if calc_stats:
		doc_ids = [doc["id"] for doc in example["rag"]["clinical_case_options"]["MedCorp"]["RRF-2"]] if "rag" in example else []
		stats = {"num_docs_in": num_docs_in, "doc_sources": doc_ids}
	return input_text, answer_text, full_text, stats

def batch(iterable: Sized, n=1) -> Iterator:
	"""
	Yield successive n-sized chunks from iterable.

	Args:
		iterable (`Sized`):
			The iterable to split.
		n (`int`, optional):
			The size of the chunks. Defaults to `1`.

	Yields:
		`Iterator`:
			An iterator with the chunks.
	"""
	l: int = len(iterable)
	p: int = math.ceil(l / n)
	for ndx in range(0, l, p):
		yield iterable[ndx : min(ndx + p, l)]


def clean_references_answer(text, example):
	"""
	Naively remove all words in text that are only present in the correct option
	:param text:
	:param example:
	:return:
	"""
	other_options = [op_text for op_i, op_text in example["options"].items() if op_i != str(example["correct_option"])]
	other_options_words = []
	correct_option_words = example["options"][str(example["correct_option"])].split()
	for option in other_options:
		other_options_words.extend(option.split())

	correct_unique_words = list(set(correct_option_words) - set(other_options_words))
	correct_unique_words.append(str(example["correct_option"]))
	for remove_word in correct_unique_words:
		if remove_word != ' ':
			text = text.replace(remove_word, MASKED)
	return text

def get_other_explanations(example):
	full_answer = example["full_answer"]
	for ans_range in  example["explanations"][str(example["correct_option"])]["char_ranges"]:
		full_answer = " ".join([full_answer[:ans_range[0]], full_answer[ans_range[1]:]])
	return full_answer


def prepare_data(
	example: Dict,
	tokenizer: PreTrainedTokenizerBase,
	is_encoder_decoder: bool = False,
	max_length: int = 2048,
	inference: bool = False,
	prompt_loss_weight: float = 0.05,
	prompt_style: str = PROMPT_01_CLEAN,
	grounding_entities: List[str] = None,
	goal: str = GOAL_ARG_GEN,
	mark_correct: bool = True,
	grounding_langs: List[str] = None,
	max_rag_docs:int = None
) -> BatchEncoding:
	"""
	Prepare data for training or inference.

	Args:
		example (`str`):
			The example to prepare.
		tokenizer (`PreTrainedTokenizerBase`):
			The tokenizer to use.
		is_encoder_decoder (`bool`, optional):
			Whether the model is an encoder-decoder model. Defaults to `False`.
		max_length (`int`, optional):
			The maximum length of the input. Defaults to `2048`.
		inference (`bool`, optional):
			Whether to prepare the data for inference. During inference labels
			are not included in model inputs. Defaults to `False`.
		prompt_loss_weight (`float`, optional):
			The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total weight
			of 5% in the loss while the result tokens will have a total weight of 95%. Defaults to `0.05`.
		prompt_style (`str`, optional):
			Which part of the sentence consider as non-prompt. Defaults to `result`.
			It can be 'results', 'sentence', 'all'.
		grounding_entities:
		goal:
		mark_correct:
		grounding_langs:
		max_rag_docs:

	Returns:
		`BatchEncoding`: `BatchEncoding` with the prepared data.

		"rag": {
        "clinical_case_options": {
            "PubMed": {
                "RRF-2": [
	"""
	input_text = ""
	answer_text = ""
	full_text = ""

	if goal == GOAL_ARG_GEN:
		raise ValueError(f"Goal '{goal}' is no longer supported. Please use '{GOAL_ANS_IDX}' instead.")
	elif goal == GOAL_ANS_IDX:
		input_text, answer_text, full_text, _ = build_prompt(example, tokenizer, prompt_style, grounding_entities,
														  grounding_langs, max_length, max_rag_docs=max_rag_docs)


	if is_encoder_decoder:
		model_inputs = tokenizer(
			text=input_text,
			max_length=max_length,
			truncation=True,
			padding=False,
			return_tensors=None,
			add_special_tokens=True,
		)
		if not inference:
			model_inputs["labels"] = tokenizer(
				text_target=answer_text,
				max_length=max_length,
				truncation=True,
				padding=False,
				return_tensors=None,
				add_special_tokens=True,
			)["input_ids"]

			model_inputs["loss_weight_mask"] = np.ones(len(model_inputs["labels"]), dtype=np.float32)

	else:
		if inference:
			model_inputs = tokenizer(
				text=input_text,
				max_length=max_length,
				truncation=True,
				padding=False,
				return_tensors=None,
				add_special_tokens=True,
			)

			# Remove the last token if it is an eos token
			if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
				model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
				model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]

		else:
			model_inputs = tokenizer(
				text=full_text,
				max_length=max_length,
				truncation=True,
				padding=False,
				return_tensors=None,
				add_special_tokens=True,
			)

			# Make sure the `eos_token_id` is added at the end
			# This bug is reported at https://github.com/huggingface/transformers/issues/22794
			if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
				model_inputs["input_ids"].append(tokenizer.eos_token_id)
				model_inputs["attention_mask"].append(1)

			model_inputs["labels"] = model_inputs["input_ids"].copy()

			# Find the prompt length
			prompt = tokenizer(
				text=input_text,
				max_length=max_length,
				truncation=True,
				padding=False,
				return_tensors=None,
				add_special_tokens=True,
			)["input_ids"]

			# Remove the last token if it is an eos token
			if prompt[-1] == tokenizer.eos_token_id:
				prompt = prompt[:-1]

			if len(prompt) > len(model_inputs["labels"]):
				raise ValueError(
					f"Prompt is longer than the labels, something went wrong. Prompt: {prompt}, labels:"
					f" {model_inputs['labels']}"
				)

			# Create the weight mask
			loss_weight_mask = np.ones(len(model_inputs["labels"]), dtype=np.float32)

			# The sum of the loss of the prompt tokens should be equal
			# to 'prompt_loss_weight' percent of the total loss
			len_prompt = len(prompt)
			len_result = len(model_inputs["labels"]) - len_prompt
			prompt_token_weight = len_result * prompt_loss_weight  # 'prompt_loss_weight' percent of the total loss
			try:
				prompt_token_weight = prompt_token_weight * (
					len_result / (len_result * (1 - prompt_loss_weight))
				)  # Scale so result tokens can have 1.0 weight
				prompt_token_weight = prompt_token_weight / len_prompt  # Divide by the number of prompt tokens
			except ZeroDivisionError:
				logging.warning(
					"Found division by zero in prompt token weight calculation. You might have an empty prompt,"
					f" empty result, or both. Example with error: {example}. Setting prompt token weight to 0.0."
				)
				prompt_token_weight = 0.0

			for i in range(len(prompt)):
				loss_weight_mask[i] = prompt_token_weight

			model_inputs["loss_weight_mask"] = loss_weight_mask

	if "token_type_ids" in model_inputs:
		# LLaMa tokenizer adds token type ids, but we don't need them
		model_inputs.pop("token_type_ids")

	return model_inputs


def batch_tokenization(
	tokenizer: PreTrainedTokenizerBase,
	dataset_name: str,
	is_encoder_decoder: bool,
	max_length: int,
	inference: bool,
	prompt_loss_weight: float,
	prompt_style: str,
	grounding_entities: List,
	goal: str,
	mark_correct: bool,
	grounding_langs: List,
	max_rag_docs:int,# <- add more args from this point on
	examples: List[Dict],
	process_no: int,
) -> List[BatchEncoding]:
	"""
	Batch tokenization function.

	Args:
		tokenizer (`PreTrainedTokenizerBase`):
			The tokenizer to use.
		dataset_name (`str`):
			The name of the dataset.
		is_encoder_decoder (`bool`):
			Whether the model is an encoder-decoder model.
		max_length (`int`):
			The maximum length of the input.
		inference (`bool`):
			Whether to prepare the data for inference. If model
			`is_encoder_decoder=False`, inputs ids will be truncated to don't include the
			results section of the example. Labels will still include the full correct
			example. If model `is_encoder_decoder=True`, this parameter is ignored.
		prompt_loss_weight (`float`):
			The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total weight
			of 5% in the loss while the result tokens will have a total weight of 95%. Defaults to `0.05`.
		prompt_style (`str`, optional):
			Which part of the sentence consider as non-prompt. Defaults to `result`.
			It can be 'results', 'text', 'all'.
		examples (`Dict`):
			The examples to tokenize.
		process_no (`int`):
			The process number.
		grounding_entities:
		goal:
		mark_correct:
		grounding_langs:
		max_rag_docs:

	Returns:
		`List[BatchEncoding]`:
			List of BatchEncoding with the prepared data.
	"""
	tokenized_examples: List[BatchEncoding] = []
	if process_no == 0:
		with Progress(
			SpinnerColumn(),
			*Progress.get_default_columns(),
			TimeElapsedColumn(),
		) as progress:
			task = progress.add_task(f"Tokenizing {dataset_name}", total=len(examples))

			for example in examples:
				tokenized_examples.append(
					prepare_data(
						example=example,
						tokenizer=tokenizer,
						is_encoder_decoder=is_encoder_decoder,
						max_length=max_length,
						inference=inference,
						prompt_loss_weight=prompt_loss_weight,
						prompt_style=prompt_style,
						grounding_entities=grounding_entities,
						goal=goal,
						mark_correct=mark_correct,
						grounding_langs=grounding_langs,
						max_rag_docs = max_rag_docs
					)
				)
				progress.update(task, advance=1)
	else:
		tokenized_examples = [
			prepare_data(
				example=example,
				tokenizer=tokenizer,
				is_encoder_decoder=is_encoder_decoder,
				max_length=max_length,
				inference=inference,
				prompt_loss_weight=prompt_loss_weight,
				prompt_style=prompt_style,
				grounding_entities=grounding_entities,
				goal=goal,
				mark_correct=mark_correct,
				grounding_langs=grounding_langs,
				max_rag_docs = max_rag_docs
			)
			for example in examples
		]

	return tokenized_examples

class CasimedicosDataset(Dataset):
	"""
	Dataset for Casimedicos.

	Args:
		tokenizer (`PreTrainedTokenizerBase`):
			The tokenizer to use.
		dataset_path (`str`):
			The path to the jsonl file containing the dataset.
		is_encoder_decoder (`bool`, optional):
			Whether the model is an encoder-decoder model. Defaults to `False`.
		max_length (`int`, optional):
			The maximum length of the input. Defaults to `2048`.
		inference (`bool`, optional):
			Whether to prepare the data for inference. If model
			`is_encoder_decoder=False`, inputs ids will be truncated to don't include
			the results section of the example. Labels will still include the full
			correct example. If model `is_encoder_decoder=True`, this parameter is
			ignored. Defaults to `False`.
		prompt_loss_weight (`float`, optional):
			The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total weight
			of 5% in the loss while the result tokens will have a total weight of 95%. Defaults to `0.05`.
		prompt_style (`str`, optional):
			Which part of the sentence consider as non-prompt. Defaults to `result`.
			It can be 'results', 'sentence', 'text'.
		num_workers (`int`, optional):
			The number of workers to use for tokenization. Defaults to
			`min(os.cpu_count(), 16)`.
		max_examples (`Optional[int]`, optional):
			The maximum number of examples to load. Defaults to `None`. If `None` all
			examples will be loaded. If `max_examples` is smaller is set we will randomly
			sample `max_examples` examples from the dataset.
	"""

	def __init__(
		self,
		tokenizer: PreTrainedTokenizerBase,
		dataset_path: str,
		is_encoder_decoder: bool = False,
		max_length: int = 2048,
		inference: bool = False,
		prompt_loss_weight: float = 0.0,
		prompt_style: str = PROMPT_01_CLEAN,
		num_workers: int = min(os.cpu_count(), 16),
		max_examples: Optional[int] = None,
		grounding_entities: List = None,
		goal: str = GOAL_ARG_GEN,
		mark_correct: bool = True,
		grounding_langs: List = None,
		max_rag_docs: int = None
	):
		self.is_encoder_decoder = is_encoder_decoder
		self.max_length = max_length
		self.inference = inference
		self.max_examples = max_examples
		self.grounding_entities = grounding_entities
		self.goal = goal
		self.mark_correct = mark_correct
		self.grounding_langs = grounding_langs
		self.prompt_style = prompt_style
		self.max_rag_docs = max_rag_docs

		if not (0.0 <= prompt_loss_weight < 1.0):
			raise ValueError(f"Prompt loss weight must be in [0, 1). Found {prompt_loss_weight}.")

		self.prompt_loss_weight = prompt_loss_weight

		try:
			self.split, self.lang, self.dataset_name, self.dataset_type, extension = os.path.basename(dataset_path).split(".")
		except ValueError:
			raise ValueError(
				f"Something is wrong with the dataset path {dataset_path}. Please check it and ensure "
				"it follows the format `split.lang.dataset_name.type.jsonl`"
			)

		# Find pre-computed epoch datasets for training
		self.dataset_dict: Dict[int, List[BatchEncoding]] = {}
		self.dataset_keys: List[int] = []
		self.current_dataset_key: int = 0

		self.dataset_dict[0] = self.compute_tokenized_examples(
			dataset_path=dataset_path,
			num_workers=num_workers,
			tokenizer=tokenizer,
		)
		self.dataset_keys.append(0)
		self.current_dataset_key = self.dataset_keys[0]

		logging.info(f"Loaded {[len(x) for x in self.dataset_dict.values()]} examples from {dataset_path}")

	def compute_tokenized_examples(
		self,
		dataset_path,
		num_workers,
		tokenizer,
	) -> List[BatchEncoding]:
		"""
		Compute the tokenized examples.

		Args:
			dataset_path (`str`):
				The path to the jsonl file containing the dataset.
			num_workers (`int`):
				The number of workers to use for tokenization.
			tokenizer (`PreTrainedTokenizerBase`):
				The tokenizer to use.

		Returns:
			`List[BatchEncoding]`:
				List of BatchEncoding with the prepared data.

		"""

		with open(dataset_path, "r", encoding="utf8") as f:
			examples = f.readlines()

		examples = [json.loads(example.strip()) for example in examples]

		if self.max_examples is not None and self.max_examples < len(examples):
			examples = random.sample(examples, self.max_examples)

		# Multithread batch tokenization is giving RecursionError: maximum recursion depth exceeded
		if num_workers <= 1:
			return batch_tokenization(
				tokenizer=tokenizer,
				dataset_name=".".join([self.split, self.lang, self.dataset_name, self.dataset_type]),
				is_encoder_decoder=self.is_encoder_decoder,
				max_length=self.max_length,
				inference=self.inference,
				prompt_loss_weight=self.prompt_loss_weight,
				prompt_style=self.prompt_style,
				grounding_entities = self.grounding_entities,
				goal = self.goal,
				mark_correct = self.mark_correct,
				grounding_langs = self.grounding_langs,
				examples=examples,
				process_no=0,
				max_rag_docs=self.max_rag_docs
			)
		else:
			tokenizer_fn = partial(
				batch_tokenization,
				tokenizer,
				".".join([self.split, self.lang, self.dataset_name, self.dataset_type]),
				self.is_encoder_decoder,
				self.max_length,
				self.inference,
				self.prompt_loss_weight,
				self.prompt_style,
				self.grounding_entities,
				self.goal,
				self.mark_correct,
				self.grounding_langs,
				self.max_rag_docs,
			)
			with Pool(num_workers) as p:
				tokenized_examples = p.starmap(
					tokenizer_fn,
					zip(batch(examples, num_workers), range(num_workers)),
				)

			return list(chain.from_iterable(tokenized_examples))

	def __len__(self) -> int:
		return len(self.dataset_dict[self.current_dataset_key])

	def __getitem__(self, idx) -> List[BatchEncoding]:
		return self.dataset_dict[self.current_dataset_key][idx].copy()

	def rotate_split(self):
		"""
		Rotate the current dataset to the next one.
		"""
		self.current_dataset_key = self.dataset_keys[
			(self.dataset_keys.index(self.current_dataset_key) + 1) % len(self.dataset_keys)
		]

		if len(self.dataset_dict) > 1:
			logging.info(
				f' Dataset {".".join([self.split, self.lang, self.dataset_name, self.dataset_type])} rotated to split'
				f" {self.current_dataset_key}"
			)

@dataclass
class DataCollatorForCasimedicos:
	"""
	Adapted from transformers.DataCollatorForSeq2Seq to handle casimedicos data.

	Data collator that will dynamically pad the inputs received, as well as the labels.

	Args:
		tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
			The tokenizer used for encoding the data.
		model ([`PreTrainedModel`]):
			The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
			prepare the *decoder_input_ids*

			This is useful when using *label_smoothing* to avoid calculating loss twice.
		padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
			Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
			among:

			- `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
			  sequence is provided).
			- `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
			  acceptable input length for the model if that argument is not provided.
			- `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
		max_length (`int`, *optional*):
			Maximum length of the returned list and optionally padding length (see above).
		pad_to_multiple_of (`int`, *optional*):
			If set will pad the sequence to a multiple of the provided value.

			This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
			7.5 (Volta).
		label_pad_token_id (`int`, *optional*, defaults to -100):
			The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
		return_tensors (`str`):
			The type of Tensor to return. Allowable values are "np", "pt" and "tf".
	"""

	tokenizer: PreTrainedTokenizerBase
	model: Optional[Any] = None
	padding: Union[bool, str, PaddingStrategy] = True
	max_length: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	label_pad_token_id: int = -100
	return_tensors: str = "pt"

	def __call__(self, features, return_tensors=None):
		if return_tensors is None:
			return_tensors = self.return_tensors
		labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
		loss_weight_mask = (
			[feature["loss_weight_mask"] for feature in features] if "loss_weight_mask" in features[0].keys() else None
		)
		# We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
		# same length to return tensors.
		if labels is not None:
			max_label_length = max(len(l) for l in labels)
			if self.pad_to_multiple_of is not None:
				max_label_length = (
					(max_label_length + self.pad_to_multiple_of - 1)
					// self.pad_to_multiple_of
					* self.pad_to_multiple_of
				)

			padding_side = self.tokenizer.padding_side
			for feature in features:
				remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
				if isinstance(feature["labels"], list):
					feature["labels"] = (
						feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
					)
				elif padding_side == "right":
					feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
				else:
					feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

		if loss_weight_mask is not None:
			max_loss_weight_mask_length = max(len(l) for l in loss_weight_mask)
			if self.pad_to_multiple_of is not None:
				max_loss_weight_mask_length = (
					(max_loss_weight_mask_length + self.pad_to_multiple_of - 1)
					// self.pad_to_multiple_of
					* self.pad_to_multiple_of
				)

			padding_side = self.tokenizer.padding_side
			for feature in features:
				remainder = [0.0 if self.label_pad_token_id == -100 else 1.0] * (
					max_loss_weight_mask_length - len(feature["loss_weight_mask"])
				)
				if isinstance(feature["loss_weight_mask"], list):
					feature["loss_weight_mask"] = (
						feature["loss_weight_mask"] + remainder
						if padding_side == "right"
						else remainder + feature["loss_weight_mask"]
					)
				elif padding_side == "right":
					feature["loss_weight_mask"] = np.concatenate([feature["loss_weight_mask"], remainder]).astype(
						np.float32
					)
				else:
					feature["loss_weight_mask"] = np.concatenate([remainder, feature["loss_weight_mask"]]).astype(
						np.float32
					)

		features = self.tokenizer.pad(
			features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors=return_tensors,
		)

		# prepare decoder_input_ids
		if (
			labels is not None
			and self.model is not None
			and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
		):
			decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
			features["decoder_input_ids"] = decoder_input_ids

		return features
