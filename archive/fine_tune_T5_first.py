import Levenshtein
import torch
from datasets import load_dataset, load_metric
import pandas as pd
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import nltk
nltk.download('punkt', download_dir='/home/user/Projects/GPT_Keylogger/T5_FirstSentences')
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator
import tiktoken
import json

torch.cuda.empty_cache()
model_checkpoint = "t5-large"
MAX_LENGTH = 80


def apply_token(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_text = [enc.decode([token]) for token in enc.encode(text)]
    list_lengths = [len(token) for token in tokenized_text]
    return list_lengths

def encoding_lengths(text):
    lst = apply_token(text)
    row = (map(str, lst))
    return "".join([" _"+c for c in row])

def make_input(special_tokens):
    return f'Translate the Special Tokens to English. \nSpecial Tokens:{special_tokens}'

def extract_first(section):
    with open(f'data/{section}.json', 'r') as file:
        data = [json.loads(line) for line in file]

    first_items = []
    for entry in data:
        paragraphs = entry['paragraphs']
        for paragraph in paragraphs:
            first_items.append(paragraph[0])
    
    
def prepare_ds():
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

    df_train.to_csv("data/train/train.csv", index=False)
    df_validation.to_csv("data/train/validation.csv", index=False)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=MAX_LENGTH)
tokenizer.add_tokens(['_' + str(i) for i in range(20)]) # comment this if you are NOT training the model for the first time


def preprocess_function(examples):
    model_inputs = tokenizer(examples["Encoding"])
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Sentence"])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_dataset():
    prepare_ds()
    datasets = load_dataset("json", data_files={"train": "data/train/train.csv", "validation": "data/train/validation.csv"})
    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    for split, split_dataset in tokenized_datasets.items():
        split_dataset.to_json(f"data/{split}.jsonl")
    tokenized_datasets = load_dataset("json", data_files={"train": "data/train/train.jsonl", "validation": "data/train/validation.jsonl"})
    return tokenized_datasets


def trainer_prepare():
    batch_size = 64
    model_checkpoint_path = "path/to/save/model"

    args = Seq2SeqTrainingArguments(
        model_checkpoint_path,
        evaluation_strategy="steps",
        eval_steps=50000,
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="steps",
        save_steps=50000,
        learning_rate=3e-5,
        generation_num_beams=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=30,
        weight_decay=0.001,
        save_total_limit=3,
        num_train_epochs=50,
        predict_with_generate=True,
        bf16=True,
        load_best_model_at_end=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, max_length=MAX_LENGTH)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    model.resize_token_embeddings(len(tokenizer))

    return args, model, data_collator


# this is for the validation set
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # Replace -100 in the labels as we can't decode them.
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Rouge Part:
    metric_rouge = load_metric("rouge")
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
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    outputs = cos(embed_preds, embed_labels)
    result["sentence-transformer"] = torch.mean(outputs, dim=0).item()
    
    result["edit_distance"] = np.mean([Levenshtein.distance(pred, label) for pred, label in zip(decoded_preds, decoded_labels)])

    # return all in one dict
    return {k: round(v, 4) for k, v in result.items()}


tokenized_datasets = preprocess_dataset()
args, model, data_collator = trainer_prepare()

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name())
else:
    print("GPU is not available. Please check your configuration.")

accelerator = Accelerator(cpu=False)
model = model.to(accelerator.device)
print("-------Device:", accelerator.device)


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
trainer.train(resume_from_checkpoint=True)
trainer.save_model("/path/to/save/model")

