import os
import sys
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import numpy as np
import re
from sentence_transformers import SentenceTransformer
import Levenshtein
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
import tiktoken  # GPT-4 tokenizer
import json


def fix_path(path: str):
    if path.endswith("/"):
        return path[:-1]
    return path


def validate_and_read_config(config_path):
    config_path = fix_path(config_path)
    required_fields = {
        "model_path": str,
        "pretrained_model": bool,
        "save_model_to_path": str,

        "preprocessed_data_path": str,
        "processed_data_path": str,
        "MAX_LENGTH": int,

        "train_conf": {
            "BATCH_SIZE": int,
            "epochs": int,
            "learning_rate": float,
            "weight_decay": float,
            "generation_num_beams": int,
            
            "logging_steps": int,
            "save_total_limit": int,
        }
    }
    
    def validate_field(config, field, expected_type):
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(config[field], expected_type):
            raise ValueError(f"Field {field} must be of type {expected_type}")

    def validate_config(config, template):
        for field, expected_type in template.items():
            if isinstance(expected_type, dict):
                validate_field(config, field, dict)
                validate_config(config[field], expected_type)
            else:
                validate_field(config, field, expected_type)
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The file '{config_path}' does not exist.")

    with open(config_path, 'r') as file:
        config = json.load(file)    
    
    validate_config(config, required_fields)
    
    return config

def _apply_token(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_text = [enc.decode([token]) for token in enc.encode(text)]
    list_lengths = [len(token) for token in tokenized_text]
    return list_lengths

def encoding_lengths(text):
    lst = _apply_token(text)
    row = (map(str, lst))
    return "".join([" _"+c for c in row])
    
def _prepare_ds_first_sentences(preprocessed_data_path: str, processed_data_path: str, MAX_LENGTH):
    preprocessed_data_path = fix_path(preprocessed_data_path)
    processed_data_path = fix_path(processed_data_path)

    def make_input(special_tokens):
        return f'Translate the Special Tokens to English. \nSpecial Tokens:{special_tokens}'

    def extract_first(section):
        file_path = os.path.join(preprocessed_data_path, f'{section}.json')
        
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load the entire file as a single JSON object
        
        first_items = []
        for entry in data['paragraphs']:  # Assuming the top-level key is 'paragraphs'
            for paragraph in entry:
                first_items.append(paragraph[0])
        
        return first_items
    
    df_validation = pd.DataFrame(columns=["Sentence"])
    df_validation["Sentence"] = extract_first("validation")
    df_validation["Encoding"] = df_validation["Sentence"].apply(encoding_lengths)
    df_validation = df_validation[df_validation["Encoding"].apply(lambda x: len(x.split(' '))) < MAX_LENGTH]
    df_validation["Encoding"] = df_validation.apply(lambda row: make_input(row["Encoding"]), axis=1)
    
    df_train = pd.DataFrame(columns=["Sentence"])
    df_train["Sentence"] = extract_first("train")
    df_train["Encoding"] = df_train["Sentence"].apply(encoding_lengths)
    df_train = df_train[df_train["Encoding"].apply(lambda x: len(x.split(' '))) < MAX_LENGTH]
    df_train["Encoding"] = df_train.apply(lambda row: make_input(row["Encoding"]), axis=1)

    df_train.to_json(f"{processed_data_path}/train.jsonl", orient="records", lines=True)
    df_validation.to_json(f"{processed_data_path}/validation.jsonl", orient="records", lines=True)


def _prepare_ds_middle_sentences(preprocessed_data_path: str, processed_data_path: str, MAX_LENGTH: int):
    preprocessed_data_path = fix_path(preprocessed_data_path)
    processed_data_path = fix_path(processed_data_path)
    def make_input(context, special_tokens):
        return f'Translate the Special Tokens to English, given the context. \nContext: {context} \nSpecial Tokens:{special_tokens}'

    for section in ("validation", "train"):
        file_path = os.path.join(preprocessed_data_path, f'{section}.json')
        
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load the entire file as a single JSON object
        
        pairs = []
        for entry in data:
            paragraphs = entry['paragraphs']
            for paragraph in paragraphs:
                for i in range(len(paragraph) - 1):
                    context = paragraph[i]
                    sentence = paragraph[i + 1]
                    pairs.append([context, sentence])

        # Convert the pairs to a DataFrame
        df_total = pd.DataFrame(pairs, columns=['Context', 'Sentence'])

        df_total["Tokens_Sentence"] = df_total["Sentence"].apply(encoding_lengths)
        df_total = df_total[df_total["Tokens_Sentence"].apply(lambda x: len(x.split(' '))) < MAX_LENGTH//2 - 10]

        df_total = df_total[df_total["Context"].apply(lambda x: len(x.split(' '))) < MAX_LENGTH//2 - 10]

        df_total["Encoding"] = df_total.apply(lambda row: make_input(row["Context"], row["Tokens_Sentence"]), axis=1)
        df_total = df_total.loc[:, ["Sentence", "Encoding"]]
        df_total.to_json(f"{processed_data_path}/{section}.jsonl", orient="records", lines=True, escapechar='\\')


def _preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(examples["Encoding"])
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Sentence"])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_dataset(first_sentences: bool, preprocessed_data_path: str, processed_data_path: str, MAX_LENGTH: int, tokenizer: AutoTokenizer):
    preprocessed_data_path = fix_path(preprocessed_data_path)
    processed_data_path = fix_path(processed_data_path)
    # if already procceesed, remove the following line:
    # print("Processing Dataset...")
    # if first_sentences:
    #     _prepare_ds_first_sentences(preprocessed_data_path, processed_data_path, MAX_LENGTH)
    # else:
    #     _prepare_ds_middle_sentences(preprocessed_data_path, processed_data_path, MAX_LENGTH)
    # dataset = load_dataset("json", data_files={"train": f"{processed_data_path}/train.jsonl", "validation": f"{processed_data_path}/validation.jsonl"})
    # tokenized_datasets = dataset.map(_preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    # for split, split_dataset in tokenized_datasets.items():
    #     split_dataset.to_json(f"{processed_data_path}/{split}.jsonl")
    # until here
    print("Loading Dataset...")
    tokenized_datasets = load_dataset("json", data_files={"train": f"{processed_data_path}/train.jsonl", "validation": f"{processed_data_path}/validation.jsonl"})
    return tokenized_datasets


def trainer_prepare(tokenizer: AutoTokenizer, model_path: str, MAX_LENGTH: int, train_conf: dict):
    batch_size = train_conf["BATCH_SIZE"]

    args = Seq2SeqTrainingArguments(
        model_path,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=train_conf["logging_steps"],
        save_strategy="epoch",
        save_total_limit=train_conf["save_total_limit"],

        learning_rate=train_conf["learning_rate"],
        generation_num_beams=train_conf["generation_num_beams"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size//2,
        weight_decay=train_conf["weight_decay"],
        num_train_epochs=train_conf["epochs"],
        predict_with_generate=True,
        bf16=True,
        load_best_model_at_end=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, max_length=MAX_LENGTH)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    model.resize_token_embeddings(len(tokenizer))

    return args, model, data_collator


def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # Replace -100 in the labels as we can't decode them.
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    def split_into_sentences(text):
        # Define the sentence enders
        sentence_enders = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        sentences = sentence_enders.split(text.strip())
        return sentences

    # Rouge expects a newline after each sentence
    decoded_preds_rouge = ["\n".join(split_into_sentences(pred)) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(split_into_sentences(label)) for label in decoded_labels]


    # Rouge Part:
    metric_rouge = evaluate.load_metric("rouge")
    result = metric_rouge.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)
    
    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    # sentence transformer part:
    model_sentence_transformers = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
    embed_preds = model_sentence_transformers.encode(decoded_preds, convert_to_tensor=True)
    embed_labels = model_sentence_transformers.encode(decoded_labels, convert_to_tensor=True)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    outputs = cos(embed_preds, embed_labels)
    result["sentence-transformer"] = torch.mean(outputs, dim=0).item()
    
    result["edit_distance"] = np.mean([(max(len(pred), len(label)) - Levenshtein.distance(pred, label)) / max(len(pred), len(label)) for pred, label in zip(decoded_preds, decoded_labels)])

    # return all in one dict
    return {k: round(v, 4) for k, v in result.items()}



def main(config_path: str, first_sentences: bool):
    # uncomment if used as a standalone script
    # if len(sys.argv) != 2:
    #     print("Usage: python generate.py <configuration file path>")
    #     sys.exit(1)
    
    # config_path = sys.argv[1]
    
    try:
        config = validate_and_read_config(config_path)

        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], model_max_length=config["MAX_LENGTH"])
        # comment this if you are NOT training the model for the first time:
        if config["pretrained_model"]:
            tokenizer.add_tokens(['_' + str(i) for i in range(20)])
        tokenized_datasets = preprocess_dataset(first_sentences, config["preprocessed_data_path"], config["processed_data_path"], config["MAX_LENGTH"], tokenizer)
        args, model, data_collator = trainer_prepare(tokenizer, config["model_path"], config["MAX_LENGTH"], config["train_conf"])

        if torch.cuda.is_available():
            print("Using GPU:", torch.cuda.get_device_name())
        else:
            print("GPU is not available. Please check your configuration.")

        accelerator = Accelerator(cpu=False)
        model = model.to(accelerator.device)
        print("-------Device:", accelerator.device, "-------")

        trainer = Seq2SeqTrainer(
                model,
                args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

        print("Start training")
        trainer.train() # add resume_from_checkpoint=True if you want to resume training exactly from last checkpoint (save)
        trainer.save_model(fix_path(config["save_model_to_path"]))


    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)    

# if __name__ == '__main__':
#     main()
