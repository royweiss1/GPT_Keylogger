import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
import os
import tiktoken
import csv
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_checkpoint = "/royweiss1/T5_FirstSentences" # from huggingface

first_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
first_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
MAX_LENGTH = 80
NUM_OF_OUTPUTS = 64
BATCH_SIZE = 8

accelerator = Accelerator(cpu=False)
torch.cuda.empty_cache()
print("Device:", accelerator.device)
first_model = first_model.to(accelerator.device)


# Load the JSONL file
with open('data/test.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

rows = []
for entry in data:
    paragraphs = entry['paragraphs']
    for paragraph in paragraphs:
        if len(paragraph) < 45:
            paragraph += [''] * (45 - len(paragraph))
        rows.append(paragraph)

# Convert the rows to a DataFrame
df = pd.DataFrame(rows)

# Rename the columns to Sentence_0, Sentence_1, ...
df.columns = [f'Sentence_{i}' for i in range(45)]

# Save to CSV
df.to_csv('data/test_splitted.csv', index=False)



def generate_first(encodings): # |encodings| = BATCH_SIZE
    all_predictions = []
    
    inputs = first_tokenizer(encodings, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")

    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    # Generate text using the model on the same device
    outputs = first_model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        output_scores=True,
        return_dict_in_generate=True,
        no_repeat_ngram_size=2,
        top_k=50,
        num_beam_groups=16,
        num_beams=NUM_OF_OUTPUTS,
        diversity_penalty=0.8,
        num_return_sequences=NUM_OF_OUTPUTS
    )

    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores
    
    # Determine the number of input examples
    batch_size = len(encodings)
    
    # Extract the highest scoring sequence for each example in the batch
    for i in range(batch_size):
        start_idx = i * NUM_OF_OUTPUTS
        end_idx = (i + 1) * NUM_OF_OUTPUTS
        
        # Get the sequences and scores for the current example
        current_sequences = sequences[start_idx:end_idx]
        current_scores = sequence_scores[start_idx:end_idx]
        
        # Find the index of the highest scoring sequence for the current example
        top_index = current_scores.argmax().item()
        
        # Extract the highest scoring sequence
        top_sequence = current_sequences[top_index]
        
        # Convert the top sequence to a readable format (e.g., string)
        top_text = first_tokenizer.decode(top_sequence, skip_special_tokens=True)
        
        # Add the top text to the list of all predictions
        all_predictions.append(top_text)

    return all_predictions


def _apply_token(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_text = [enc.decode([token]) for token in enc.encode(text)]
    list_lengths = [len(token) for token in tokenized_text]
    return list_lengths


def encoding_lengths(text):
    lst = _apply_token(text)
    row = (map(str, lst))
    return "".join([" _"+c for c in row])


def make_input(special_tokens):
    return f'Translate the Special Tokens to English. \nSpecial Tokens:{special_tokens}'


def decode_text(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    decoded_tokens = [enc.decode([token]) for token in text]
    return decoded_tokens


def encode_text(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_text = enc.encode(text)
    return tokenized_text

def create_batches(lst):
    return [lst[i:i + BATCH_SIZE] for i in range(0, len(lst), BATCH_SIZE)]


def generate_first_custom(csv_file_path="data/test_splitted.csv", checkpoint_interval=50):
    # Load the test set
    test_set = pd.read_csv("data/test_splitted.csv")
    all_first_sentences = list(test_set["Sentence_0"])
    first_sentences_inputs = []

    for sentence in all_first_sentences:
        if str(sentence) != "nan":
            first_sentences_inputs.append(make_input(encoding_lengths(sentence)))
    
    batch_split = create_batches(first_sentences_inputs)
    
    batch_count = 0
    generated_results = ["Generated_0"]    
    for batch in tqdm(batch_split[batch_count:], desc="batch progress: "):  # batch is a list
        sentences_generated = generate_first(batch)
        for index, sentences in enumerate(sentences_generated):
            original_index = batch_count * BATCH_SIZE + index
            generated = sentences  # Assuming the top prediction is the first one
            generated_results.append(generated)

        batch_count += 1
        
        # Save checkpoint
        if batch_count % checkpoint_interval == 0:
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for item in generated_results:
                    writer.writerow([item])
            print(f"Checkpoint saved at batch {batch_count}")

    # Save the final results to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in generated_results:
            writer.writerow([item])
    print(f"Results saved to {csv_file_path}")

generate_first_custom()

torch.cuda.empty_cache()
accelerator = Accelerator(cpu=False)
print("Device:", accelerator.device)

context_model_checkpoint = "royweiss1/GPT_Keylogger_Dataset"

context_model = AutoModelForSeq2SeqLM.from_pretrained(context_model_checkpoint)
context_tokenizer = AutoTokenizer.from_pretrained(context_model_checkpoint)

context_model = context_model.to(accelerator.device)

MAX_LENGTH = 150
NUM_OF_OUTPUTS = 30
BATCH_SIZE = 8
Filter = 16

def generate_with_context(encodings):
    inputs = context_tokenizer(encodings, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
    
    all_predictions = []
    batch = {key: value.to(accelerator.device) for key, value in inputs.items()}
    outputs = context_model.generate(**batch,
                                     max_length=MAX_LENGTH,
                                     output_scores=True,
                                     return_dict_in_generate=True,
                                     length_penalty=2.0,
                                     no_repeat_ngram_size=2,
                                     top_k=50,
                                     num_beam_groups=5,
                                     num_beams=NUM_OF_OUTPUTS,
                                     diversity_penalty=0.2,
                                     num_return_sequences=NUM_OF_OUTPUTS)
    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores
    
    # Determine the number of input examples
    batch_size = 1
    
    # Extract the highest scoring sequence for each example in the batch
    for i in range(batch_size):
        start_idx = i * NUM_OF_OUTPUTS
        end_idx = (i + 1) * NUM_OF_OUTPUTS
        
        # Get the sequences and scores for the current example
        current_sequences = sequences[start_idx:end_idx]
        current_scores = sequence_scores[start_idx:end_idx]
        
        # Find the index of the highest scoring sequence for the current example
        top_index = current_scores.argmax().item()
        
        # Extract the highest scoring sequence
        top_sequence = current_sequences[top_index]
        
        # Convert the top sequence to a readable format (e.g., string)
        top_text = context_tokenizer.decode(top_sequence, skip_special_tokens=True)
        
        # Add the top text to the list of all predictions
        all_predictions.append(top_text)

    return all_predictions[0]



def make_input_with_context(special_tokens, context):
    return f'Translate the Special Tokens to English, given the context. \nContext: {context} \nSpecial Tokens:{special_tokens}'


def generate_first_custom_with_context(start_idx, end_idx=None, csv_file_path="data/test_splitted.csv"):
    # Load the test set
    test_set = pd.read_csv("data/test_splitted.csv")

    if end_idx is None:
        end_idx = len(test_set)
        
    # Initialize the 'Generated' columns if they don't exist
    for i in range(len(test_set.columns) - 3):
        if not any(test_set.columns.str.startswith(f"Generated_{i}")):
            test_set[f"Generated_{i}"] = ""

    for idx in tqdm(range(start_idx, end_idx), desc="Processing rows: "):
        previous_generated = test_set.at[idx, "Generated_0"] # mind this should be Generated_0
        for i in range(1, len(test_set.columns)):
            sentence_col = f"Sentence_{i}"
            generated_col = f"Generated_{i}"

            if sentence_col in test_set.columns and str(test_set.at[idx, sentence_col]) != "nan":
                current_sentence = test_set.at[idx, sentence_col]
                input_data = make_input_with_context(encoding_lengths(current_sentence), previous_generated)
                generated_sentences = generate_with_context(input_data)

                if len(generated_sentences) > 0:
                    generated_sentence = generated_sentences
                    previous_generated = generated_sentence
                    test_set.at[idx, generated_col] = generated_sentence


        # Save the updated DataFrame to the CSV file after each row
        test_set.to_csv(f"{csv_file_path.split('.')[0]}_{start_idx}.csv", index=False)

    # Save the final results to the CSV file
    test_set.to_csv(f"{csv_file_path.split('.')[0]}_{start_idx}.csv", index=False)
    print(f"Results saved to {csv_file_path.split('.')[0]}_{start_idx}.csv")
    

generate_first_custom_with_context(0)
    
# now go to evaluate.ipynb

