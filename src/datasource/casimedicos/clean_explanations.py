import os
from openai import OpenAI
import pickle
import tiktoken
import random

from datasource.utils import load_jsonl, save_jsonl

OUT_DIR = "./out/openai/"
PROMPT = "Without changing the text, remove any direct references to the correct answer in the following explanation. Replace any direct references with the [HIDDEN] tag:"

def get_last_index():
    files = next(os.walk(OUT_DIR))[2]
    return max([int(x.replace(".pickle", "")) for x in files])

def gen_file_path(index:int):
    new_file_name = f"{index:07d}.pickle"
    return os.path.join(OUT_DIR, new_file_name)

def save_response(response):
    last_index = get_last_index()

    file_path = gen_file_path(last_index + 1)

    with open(file_path, 'wb') as handle:
        pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_response(index:int=None):
    index = get_last_index() if index is None else index
    file_path = gen_file_path(index)
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def setup_openai() -> OpenAI:
    return OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

# "Without changing the test, remove the direct reference to the correct answer in the following explanation. Replace the direct reference with a [HIDDEN] tag: Therefore, the correct answer to this question is option 3. Patients on pharmacological treatment for OP should use calcium and vitamin D supplements because practically all clinical trials that have demonstrated efficacy of antiosteoporotic drugs routinely include calcium supplements and cholecalciferol (vitamin D3), but not in monotherapy.",

def build_message(content:str):
    return {"role": "user", "content": content}

def query_openai(client: OpenAI, content:str, model:str="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        messages=[
            build_message(content)
        ],
        model=model,
        temperature=0,
    )
    save_response(response)
    return response

def gen_query(example:dict) -> str:
    options = " ".join([f"{idx}. {text}" for idx, text in example["options"].items() if isinstance(text, str)])
    incorrect_option_numbers = [1, 2, 3, 4, 5] if isinstance(example["options"]["5"], str) else [1, 2, 3, 4]
    incorrect_option_numbers.remove(example["correct_option"])
    """
    return (f"These are the options for a multiple choice question: {options} In the following explanation, and without changing"
     f" the text, remove any direct and clear references that indicate that option {example['correct_option']} is the "
     f"correct answer (if there are any). Replace any direct references with the [HIDDEN] tag. Explanation: {example['full_answer']}")
    """
    """
    return (f"In the following explanation, and without changing the text, remove any text (if there is any) that clearly"
            f" points at the correct option is \"{example['correct_option']}. {example['options'][str(example['correct_option'])]}\". "
            f"Replace any direct references with the [HIDDEN] tag.\n{example['full_answer']}")
    """
    """
    return (f"The following explanation was extracted from a multiple-choice medical question. You need to remove any "
            f"blatant pointers to the correct answer and keep the rest of the argumentation intact without changing the "
            f"original text. Replace with the [HIDDEN] tag any text of the explanation (if there is any) that clearly "
            f"indicates that the correct option is option {example['correct_option']} as well as any text that clearly "
            f"discards other options. In both cases, you must retain the argumentation. Here is the explanation:\n{example['full_answer']}")
    """
    """
    return (
        f"In the following explanation, and without changing the text, remove any direct and clear references that "
        f"indicate that option {example['correct_option']} is the correct answer (if there are any) as well as any text that clearly "
        f"discards other options. Replace any direct references to the options with the [HIDDEN] tag. Explanation: {example['full_answer']}")
    """
    """
    return (
        f"These are the options for a multiple choice question: {options} In the following explanation, and without changing"
        f" the text, remove any direct and clear references that indicate that option {example['correct_option']} is the "
        f"correct answer (if there are any). Replace any direct references with the [HIDDEN] tag. Explanation: {example['full_answer']}")
    """
    """
    return (f'In the following text, remove all references that clearly state that any of the options 1, 2, 3, {"4 or 5" if isinstance(example["options"]["5"], str) else "or 4"} are '
            f'eiter correct or false. Do not change the original text, only replace with the tag [HIDDEN] the references '
            f'to the correct or incorrect option if there are any. For example; the text "option 4 is correct" should be '
            f'"option [HIDDEN] is correct", or "{example["options"][str(example["correct_option"])]} is the right answer" '
            f'should be "[HIDDEN] is the [HIDDEN] answer": {example["full_answer"]}')
    """

    # EN
    return (f'In the following text, remove all references that clearly state that any of the options 1, 2, 3, '
            f'{"4 or 5" if isinstance(example["options"]["5"], str) else "or 4"} are either correct or false. Don\'t '
            f'change the original text and don\'t write linebreaks; only replace with the tag [HIDDEN] the text that '
            f'says that something is the correct or incorrect option if there is are any. Don\'t replace text that '
            f'doesn\'t specifically imply that certain something is the right or wrong answer. For example: '
            f'the text "option {str(example["correct_option"])} is correct." should be "[HIDDEN]", the '
            f'text "Option {random.choice(incorrect_option_numbers)} is less likely because this and that" should '
            f'be "[HIDDEN] this and that", the text "answer blablabla is the right answer because whatever" should '
            f'be "answer blablabla is [HIDDEN] whatever". Here is the text: {example["full_answer"]}')

    #return f'En el siguiente texto médico, elimina todas las referencias que indiquen claramente que cualquiera de las opciones 1, 2, 3{", 4 or 5" if isinstance(example["options"]["5"], str) else " or 4"} es correcta o falsa. No cambies el texto original y no escribas saltos de línea; únicamente sustituye por el tag [HIDDEN] aquel texto que diga que algo es la opción correcta o incorrecta (si es que existe dicho texto). No sustituyas texto que no menciona específicamente que algo es la respuesta correcta o incorrecta. Por ejemplo: el texto "la opción {str(example["correct_option"])} es la correcta." debería ser "[HIDDEN]", el texto "La opción {random.choice(incorrect_option_numbers)} es menos probable porque tal y cual" debería ser "[HIDDEN] tal y cual", el texto "la respuesta X es la respuesta correcta porque Y" debería ser "la respuesta X es [HIDDEN] Y". Aplica esta instrucción al siguiente texto: {example["full_answer"]}'
    #return f'En la siguiente respuesta de un examen de medicina, elimina cualquier texto que indique claramente cual de las opciones 1, 2, 3{", 4 o 5" if isinstance(example["options"]["5"], str) else " o 4"} es correcta y cuales incorrectas. No cambies el texto original y no escribas saltos de línea; únicamente sustituye por el tag [HIDDEN] aquel texto que diga que algo es la opción correcta o incorrecta (si es que existe dicho texto). Por ejemplo: el texto "la opción {str(example["correct_option"])} es la correcta." debería ser "[HIDDEN]", el texto "La opción {random.choice(incorrect_option_numbers)} es menos probable porque tal y cual" debería ser "[HIDDEN] tal y cual", el texto "la respuesta X es la respuesta correcta porque Y" debería ser "la respuesta X es [HIDDEN] Y". Aplica esta instrucción al siguiente texto: {example["full_answer"]}'
    #return f'En la siguiente respuesta de un examen de medicina, elimina cualquier texto que indique claramente cual de las opciones 1, 2, 3{", 4 o 5" if isinstance(example["options"]["5"], str) else " o 4"} es correcta y cuales incorrectas. No cambies el texto original y no escribas saltos de línea; únicamente sustituye por el tag [HIDDEN] aquel texto que diga que algo es la opción correcta o incorrecta (si es que existe dicho texto). No sustituyas texto que no haga referencia a la opción correcta o incorrecta. Por ejemplo: el texto "la opción {str(example["correct_option"])} es la correcta." debería ser "[HIDDEN]", el texto "La opción {random.choice(incorrect_option_numbers)} es menos probable porque tal y cual" debería ser "[HIDDEN] tal y cual", el texto "la respuesta X es la respuesta correcta porque Y" debería ser "la respuesta X es [HIDDEN] Y", el texto "tengo dudas entre las opciones {str(example["correct_option"])} y {random.choice(incorrect_option_numbers)} porque X" debería ser "[HIDDEN] X". Aplica esta instrucción al siguiente texto: {example["full_answer"]}'


def num_tokens_from_prompt(content, model="gpt-3.5-turbo-0613"):
    message = build_message(content)
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_prompt(content, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_prompt(content, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    num_tokens += tokens_per_message
    for key, value in message.items():
        num_tokens += len(encoding.encode(value))
        if key == "name":
            num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def num_tokens_from_completion(response:str, model:str = "gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(response))

def count_tokens(query, completion, model:str="gpt-3.5-turbo"):
    prompt_tokens = num_tokens_from_prompt(query, model)
    completion_tokens = num_tokens_from_completion(completion, model)
    #print(f"{prompt_tokens} + {completion_tokens} = {prompt_tokens + completion_tokens}")
    return prompt_tokens, completion_tokens

def calculate_cost(model_prompt_tokens, model_completion_tokens, model):
    if model in {"gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-1106-vision-preview"}:
        rate_prompt = 0.01
        rate_completion = 0.03
    elif model == "gpt-4":
        rate_prompt = 0.03
        rate_completion = 0.06
    elif model == "gpt-4-32k":
        rate_prompt = 0.06
        rate_completion = 0.12
    elif model in {"gpt-3.5-turbo-0125", "gpt-3.5-turbo"}:
        rate_prompt = 0.0005
        rate_completion = 0.0015
    elif model == "gpt-3.5-turbo-instruct":
        rate_prompt = 0.0015
        rate_completion = 0.0020
    else:
        raise NotImplementedError(
            f"""calculate_cost() is not implemented for model {model}. See https://openai.com/pricing for model rates."""
        )
    return (model_prompt_tokens*rate_prompt)/1000 + (model_completion_tokens*rate_completion)/1000

def calculate_budget(model:str="gpt-3.5-turbo"):
    languages = ['it','fr']#['es', 'en', 'it', 'fr']
    modes = ['train', 'test', 'dev']#['train', 'dev', 'test']
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for lang in languages:
        for mode in modes:
            model_prompt_tokens = 0
            model_completion_tokens = 0
            dataset_path = f"./data/casimedicos/{mode}.{lang}.casimedicos.grounded.jsonl"
            examples = load_jsonl(dataset_path, False)
            for example in examples:
                prompt_tokens, completion_tokens = count_tokens(gen_query(example), example["full_answer"], model)
                model_prompt_tokens += prompt_tokens
                model_completion_tokens += completion_tokens
            cost = calculate_cost(model_prompt_tokens, model_completion_tokens, model)
            print(f"{lang} {mode}: {model_prompt_tokens} + {model_completion_tokens} = {model_prompt_tokens + model_completion_tokens} -> {cost:.2f}$")
            total_prompt_tokens += model_prompt_tokens
            total_completion_tokens += model_completion_tokens
    cost = calculate_cost(total_prompt_tokens, total_completion_tokens, model)
    print(f"Total: {total_prompt_tokens} + {total_completion_tokens} = {total_prompt_tokens + total_completion_tokens} -> {cost:.2f}$")

def run(model:str):
    languages = ['es', 'en', 'it', 'fr']  # ['es', 'en', 'it', 'fr']
    modes = ['train', 'dev', 'test']
    total_prompt_tokens = 0
    total_completion_tokens = 0
    client = setup_openai()
    for lang in languages:
        for mode in modes:
            model_prompt_tokens = 0
            model_completion_tokens = 0
            updated_examples = []
            dataset_path = f"./data/casimedicos/{mode}.{lang}.casimedicos.grounded.jsonl"
            examples = load_jsonl(dataset_path, True)
            counter = 0
            for example_id, example in examples.items():
                query = gen_query(example)
                response = query_openai(client, content=query, model=model)
                response_content = response.choices[0].message.content
                print(f'{{"example_id":"{example_id}", "original":"{example["full_answer"]}", "response":"{response_content}"}}')
                example["full_answer_no_ref"] = response_content
                updated_examples.append(example)
                # Cost calculation
                prompt_tokens, completion_tokens = count_tokens(gen_query(example), response_content, model)
                model_prompt_tokens += prompt_tokens
                model_completion_tokens += completion_tokens
                counter += 1
                if counter >= 15 and False:
                    break
            save_jsonl(f"./data/casimedicos/{mode}.{lang}.casimedicos.clean.jsonl", updated_examples)
            cost = calculate_cost(model_prompt_tokens, model_completion_tokens, model)
            print(f"{lang} {mode}: {model_prompt_tokens} + {model_completion_tokens} = {model_prompt_tokens + model_completion_tokens} -> {cost:.2f}$")
            total_prompt_tokens += model_prompt_tokens
            total_completion_tokens += model_completion_tokens
    cost = calculate_cost(total_prompt_tokens, total_completion_tokens, model)
    print(
        f"Total: {total_prompt_tokens} + {total_completion_tokens} = {total_prompt_tokens + total_completion_tokens} -> {cost:.2f}$")

if __name__ == '__main__':
    calculate_budget("gpt-4-1106-preview")
    #run("gpt-4-0125-preview")
    #run("gpt-4-1106-preview")
    #run("gpt-3.5-turbo-0125")
    #calculate_budget("gpt-3.5-turbo")
    #calculate_budget("gpt-3.5-turbo-0125")