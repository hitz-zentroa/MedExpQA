from tqdm import tqdm
import os
import json
import re
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import argparse

SPLIT_PROMPT = {
    "es": "La opción correcta es: ",
    "en": "The correct answer is: ",
    "it": "L'opzione corretta è: ",
    "fr": "La bonne option est: "
}

def acc_eval(pred_path:str, lang_dataset:str):#, lang_grounding:str, model_name:str, checkpoint:str):
    predictions = []
    references = []
    for file in os.listdir(pred_path):
        if file.endswith(".predictions.jsonl"):
            pred_path = os.path.join(pred_path, file)
    with open(pred_path, "r", encoding="utf8") as f:
        examples = json.load(f)
        for example_dict in tqdm(examples, desc="Prediction evaluation"):
            #predictions.append(example_dict["prediction"].replace(f"{STRINGS_MULTI_LANG[PROMPT_SEP_DEFAULT][lang_dataset]}: ","")[0])
            if len(example_dict["prediction"].split(SPLIT_PROMPT[lang_dataset])) < 2:
                print(f"Rare case where prediction doesn't contain prompt_sep in {pred_path} ({pred_path.split('_')[-2]})")
                predictions.append("x")
            else:
                predictions.append(example_dict["prediction"].split(SPLIT_PROMPT[lang_dataset])[1][0])
            #predictions.append(example_dict["prediction"][-1])
            references.append(example_dict["gold_answer"][0])
    acc = sum(1 for x,y in zip(predictions,references) if x == y) / len(predictions)
    #print(f"{model_name}:{checkpoint} {lang_dataset}_{lang_grounding} acc: {acc*100:.1f}")
    return acc

def evaluate_folder(path, include_root:bool = False, for_scale:bool = False):
    #d = {'model': [], 'checkpoint': [], 'dataset_lang': [], 'grounding_lang': [], 'acc': []}
    data = []
    sub_paths = [f.path for f in os.scandir(path) if f.is_dir()]
    if include_root:
        sub_paths.append(path)
    for sub_path in sub_paths:
        split_path = sub_path.split('/')
        if "checkpoint" in split_path[-1]:
            checkpoint = split_path[-1].split("-")[-1]
            exp_folder = split_path[-2]
        else:
            checkpoint = "999999"
            exp_folder = split_path[-1]
        model_name = exp_folder.split('_')[1]
        if for_scale:
            grounding = exp_folder.split('_')[-1]
            grounding = grounding if grounding != 'none' else '0'
        else:
            grounding = exp_folder.split('_')[3] #if exp_folder.split('_')[2] == '7b' else exp_folder.split('_')[2]
        lang_dataset, lang_grounding = re.findall(r'(es|en|it|fr)_.*(es|en|multi|none)', exp_folder)[0]
        #pred_path = f"{sub_path}/test.{lang_dataset}.casimedicos.grounded.predictions.jsonl"
        #pred_path = f"{sub_path}/dev_test.{lang_dataset}.casimedicos.disorders.predictions.jsonl"
        #pred_path = f"{sub_path}/test.{lang_dataset}.casimedicos.clean.predictions.jsonl"
        acc = acc_eval(pred_path=sub_path, lang_dataset=lang_dataset)
        data.append([model_name, grounding, int(checkpoint), lang_dataset, lang_grounding, acc])
    return data

def plot_results(df:pd.DataFrame):
    sns.lineplot(data=df, x='checkpoint', y='acc',
                 hue=df[['model', 'grounding', 'dataset_lang', 'grounding_lang']].apply(tuple, axis=1))
    plt.show()

def load_paths_from_file(path_file:str):
    paths = []
    with open(path_file) as file:
        paths = [line.rstrip() for line in file]
    return paths

def run(path_file:str, include_root:bool=False, plot_chart:bool=False):
    paths = load_paths_from_file(path_file)
    data = []
    for path in paths:
        data.extend(evaluate_folder(path, include_root=include_root))
    df = pd.DataFrame(data, columns=['model', 'grounding', 'checkpoint', 'dataset_lang', 'grounding_lang', 'acc'])
    if plot_chart:
        plot_results(df)
    # Get only the rows from the last checkpoint for each model config (change checkpoint with acc to get max acc
    df = df[df.groupby(['model', 'grounding', 'dataset_lang', 'grounding_lang'])['acc'].transform(max) == df['acc']]
    df = df.sort_values(by=['model', 'dataset_lang', 'grounding_lang'])

    for index, row in df.iterrows():
        print(
            f"{row['model']}({row['grounding']})_{row['dataset_lang']}_{row['grounding_lang']}: {row['acc'] * 100:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", help="Path to .txt file with the list of directories to evaluate.", type=str)
    args = parser.parse_args()
    run(args.pred_path)
    #plot_doc_scale()