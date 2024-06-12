import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
import os
import tiktoken  # GPT-4 tokenizer
import csv
import json
import sys

import scripts.evaluate_script as evaluate_script


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def validate_and_read_config(config_path):
    required_fields = {
        "test_dataset_path": str,
        "generated_output_path": str,
        "BATCH_SIZE": int,

        "evaluate": bool,
        "generated_metrics_path": str,

        "first_sentences_generation_config": {
            "first_sentences_model_name": str,
            "MAX_LENGTH": int,
            "top_k": int,
            "num_beam_groups": int,
            "num_beams": int,

            "no_repeat_ngram_size": int,
            "diversity_penalty": (int, float)
        },
        "middle_sentences_generation_config": {
            "middle_sentences_model_name": str,
            "MAX_LENGTH": int,
            "top_k": int,
            "num_beam_groups": int,
            "num_beams": int,
            "length_penalty": (int, float),
            "no_repeat_ngram_size": int,
            "diversity_penalty": (int, float)
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


def data_process(test_dataset_path: str) -> pd.DataFrame:
    if not test_dataset_path.endswith('.json') and not test_dataset_path.endswith('.jsonl'):
        if '.' in test_dataset_path:
            raise ValueError("The test dataset must be a JSONL file.")
        test_dataset_path += '.jsonl'

    # Load the JSONL file
    with open(test_dataset_path, 'r') as file:
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
    return df


def _generate_first(encodings: list[str], model_name: str, MAX_LENGTH: int, top_k: int, num_beam_groups: int,  NUM_OF_OUTPUTS: int, no_repeat_ngram_size: int, diversity_penalty: float): # |encodings| = BATCH_SIZE

    first_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    first_tokenizer = AutoTokenizer.from_pretrained(model_name)

    accelerator = Accelerator(cpu=False)
    torch.cuda.empty_cache()
    print("--------- Device:", accelerator.device, "---------")
    first_model = first_model.to(accelerator.device)

    all_predictions = []
    
    inputs = first_tokenizer(encodings, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")

    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    # Generate text using the model on the same device
    outputs = first_model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        output_scores=True,
        return_dict_in_generate=True,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_k=top_k,
        num_beam_groups=num_beam_groups,
        num_beams=NUM_OF_OUTPUTS,
        diversity_penalty=diversity_penalty,
        num_return_sequences=NUM_OF_OUTPUTS
    )

    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores
    
    batch_size = len(encodings)
    
    for i in range(batch_size):
        start_idx = i * NUM_OF_OUTPUTS
        end_idx = (i + 1) * NUM_OF_OUTPUTS
        
        current_sequences = sequences[start_idx:end_idx]
        current_scores = sequence_scores[start_idx:end_idx]
        
        top_index = current_scores.argmax().item()
        top_sequence = current_sequences[top_index]
        top_text = first_tokenizer.decode(top_sequence, skip_special_tokens=True)
        
        all_predictions.append(top_text)

    return all_predictions



def _apply_token(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_text = [enc.decode([token]) for token in enc.encode(text)]
    list_lengths = [len(token) for token in tokenized_text]
    return list_lengths

def _encoding_lengths(text):
    lst = _apply_token(text)
    row = (map(str, lst))
    return "".join([" _"+c for c in row])

def make_input(sentence):
    return f'Translate the Special Tokens to English. \nSpecial Tokens:{_encoding_lengths(sentence)}'

def create_batches(lst: list, BATCH_SIZE: int):
    return [lst[i:i + BATCH_SIZE] for i in range(0, len(lst), BATCH_SIZE)]


def generate_first_custom(test_set: pd.DataFrame, generated_results_path: str, generate_config: dict, checkpoint_interval=50):
    # Load the test set
    all_first_sentences = list(test_set["Sentence_0"])
    first_sentences_inputs = []

    for sentence in all_first_sentences:
        if str(sentence) != "nan":
            first_sentences_inputs.append(make_input(sentence))
    
    batch_split = create_batches(first_sentences_inputs)
    
    batch_count = 0
    generated_results = ["Generated_0"]    
    for batch in tqdm(batch_split[batch_count:], desc="batch progress: "):  # batch is a list
        sentences_generated = _generate_first(batch, **generate_config)
        for index, sentences in enumerate(sentences_generated):
            generated_results.append(sentences[index])

        batch_count += 1
        
        # Save checkpoint
        if batch_count % checkpoint_interval == 0:
            with open(generated_results_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for item in generated_results:
                    writer.writerow([item])
            print(f"First Sentences Checkpoint saved at batch {batch_count} to {generated_results_path}")
            generated_results.clear()  # Clear the list after writing to avoid duplicates

    # Save the final results to the CSV file
    with open(generated_results_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in generated_results:
            writer.writerow([item])
    print(f"First Sentences Generated saved to {generated_results_path}")



def _generate_with_context(encodings: list[str], context_model_name: str, MAX_LENGTH: int, top_k: int, num_beam_groups: int, NUM_OF_OUTPUTS: int, length_penalty: float, no_repeat_ngram_size: int, diversity_penalty: float):
    torch.cuda.empty_cache()
    accelerator = Accelerator(cpu=False)
    print("--------- Device:", accelerator.device, "---------")

    context_model = AutoModelForSeq2SeqLM.from_pretrained(context_model_name)
    context_tokenizer = AutoTokenizer.from_pretrained(context_model_name)
    context_model = context_model.to(accelerator.device)

    inputs = context_tokenizer(encodings, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
    
    batch = {key: value.to(accelerator.device) for key, value in inputs.items()}
    outputs = context_model.generate(**batch,
                        max_length=MAX_LENGTH,
                        output_scores=True,
                        return_dict_in_generate=True,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        top_k=top_k,
                        num_beam_groups=num_beam_groups,
                        num_beams=NUM_OF_OUTPUTS,
                        diversity_penalty=diversity_penalty,
                        num_return_sequences=NUM_OF_OUTPUTS)
    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores

    top_index = sequence_scores.argmax().item()
    top_sequence = sequences[top_index]
    top_text = context_tokenizer.decode(top_sequence, skip_special_tokens=True)
    
    return top_text


def make_input_with_context(sentence, context):
    return f'Translate the Special Tokens to English, given the context. \nContext: {context} \nSpecial Tokens:{_encoding_lengths(sentence)}'


def generate_first_custom_with_context(test_set: pd.DataFrame, generated_output_path: str, generate_config: dict, start_idx=0, end_idx=None):
    # Load the test set
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
                input_data = make_input_with_context(current_sentence, previous_generated)
                generated_sentence = _generate_with_context(input_data, **generate_config)
                if len(generated_sentence) > 0: # if not blank
                    previous_generated = generated_sentence
                    test_set.at[idx, generated_col] = generated_sentence
        # After each paragraph, save the results to the CSV file
        test_set.to_csv(generated_output_path, index=False)

        test_set.to_csv(generated_output_path, index=False)
    print(f"Results saved to {generated_output_path}")



def main(config_path: str):
    # uncomment if used as a standalone script
    # if len(sys.argv) != 2:
    #     print("Usage: python generate.py <configuration file path>")
    #     sys.exit(1)
    
    # config_path = sys.argv[1]
    
    try:
        config = validate_and_read_config(config_path)
        test_set = data_process(config["test_dataset_path"])
        generate_first_custom(test_set, config["generated_output_path"])
        generate_first_custom_with_context(test_set, config["generated_results_path"], config["middle_sentences_generation_config"])
        if config["evaluate"]:
            evaluate_script.main(config["generated_output_path"], config["generated_metrics_path"])

    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)    

# if __name__ == '__main__':
#     main()

