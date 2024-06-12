import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
import os
import tiktoken  # GPT-4 tokenizer
import csv
import json
import sys

import scripts.evaluate_script as evaluate_script
logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def validate_and_read_config(config_path):
    required_fields = {
        "test_dataset_path": str,
        "generated_path": str,
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

    with open(test_dataset_path, 'r') as file:
        data = json.load(file)

    paragraphs = data['paragraphs']
    
    rows = []
    for paragraph in paragraphs:
        if len(paragraph) < 45:
            paragraph += [''] * (45 - len(paragraph))
        else:
            paragraph = paragraph[:45]
        rows.append(paragraph)

    # Convert the rows to a DataFrame
    df = pd.DataFrame(rows)
    

    # Rename the columns to Sentence_0, Sentence_1, ...
    df.columns = [f'Sentence_{i}' for i in range(45)]
    new_columns = [f'Generated_{i}' for i in range(45)]
    for col in new_columns:
        df[col] = ""

    # Save to CSV
    return df


def _generate_first(encodings: list[str], first_sentences_model_name: str, MAX_LENGTH: int, top_k: int, num_beam_groups: int,  num_beams: int, no_repeat_ngram_size: int, diversity_penalty: float): # |encodings| = BATCH_SIZE

    first_model = AutoModelForSeq2SeqLM.from_pretrained(first_sentences_model_name)
    first_tokenizer = AutoTokenizer.from_pretrained(first_sentences_model_name)

    accelerator = Accelerator(cpu=False)
    torch.cuda.empty_cache()
    # print("--------- Device:", accelerator.device, "---------")
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
        num_beams=num_beams,
        diversity_penalty=diversity_penalty,
        num_return_sequences=num_beams
    )

    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores
    
    batch_size = len(encodings)
    
    for i in range(batch_size):
        start_idx = i * num_beams
        end_idx = (i + 1) * num_beams
        
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


def generate_first_custom(test_set: pd.DataFrame, batch_size: int, generated_path: str, generate_config: dict, checkpoint_interval=50):
    # Load the test set
    test_set = test_set[test_set['Sentence_0'] != "nan"]
    test_set.to_csv(generated_path, index=False)

    all_first_sentences = list(test_set["Sentence_0"])
    first_sentences_inputs = []

    for sentence in all_first_sentences:
        first_sentences_inputs.append(make_input(sentence))
    
    batch_split = create_batches(first_sentences_inputs, batch_size)
    
    batch_count = 0
    generated_results = []

    for batch in tqdm(batch_split[batch_count:], desc="batch progress: "):  # batch is a list
        sentences_generated = _generate_first(batch, **generate_config)
        generated_results += sentences_generated  # list batch_size long
    
        batch_count += 1
        if batch_count % checkpoint_interval == 0:
            test_set['Generated_0'] = generated_results[:len(test_set)]
            test_set.to_csv(generated_path, index=False)
            print(f"Checkpoint saved at {generated_path}")
    
    if batch_count % checkpoint_interval != 0:
        test_set['Generated_0'] = generated_results[:len(test_set)]
        test_set.to_csv(generated_path, index=False)
        print(f"Final results saved at {generated_path}")



def _generate_with_context(encodings: list[str], middle_sentences_model_name: str, MAX_LENGTH: int, top_k: int, num_beam_groups: int, num_beams: int, length_penalty: float, no_repeat_ngram_size: int, diversity_penalty: float):
    torch.cuda.empty_cache()
    accelerator = Accelerator(cpu=False)
    # print("--------- Device:", accelerator.device, "---------")

    context_model = AutoModelForSeq2SeqLM.from_pretrained(middle_sentences_model_name)
    context_tokenizer = AutoTokenizer.from_pretrained(middle_sentences_model_name)
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
                        num_beams=num_beams,
                        diversity_penalty=diversity_penalty,
                        num_return_sequences=num_beams)
    sequences = outputs.sequences
    sequence_scores = outputs.sequences_scores

    top_index = sequence_scores.argmax().item()
    top_sequence = sequences[top_index]
    top_text = context_tokenizer.decode(top_sequence, skip_special_tokens=True)
    
    return top_text


def make_input_with_context(sentence, context):
    return f'Translate the Special Tokens to English, given the context. \nContext: {context} \nSpecial Tokens:{_encoding_lengths(sentence)}'


def generate_first_custom_with_context(generated_output_path: str, generate_config: dict, start_idx=0, end_idx=None):
    # Load the test set
    test_set = pd.read_csv(generated_output_path)
    
    if end_idx is None:
        end_idx = len(test_set)
        
    for i in range(len(test_set.columns) // 2):
        generated_col = f"Generated_{i}"
        if generated_col in test_set.columns:
            test_set[generated_col] = test_set[generated_col].astype(str)
    
    for idx in tqdm(range(start_idx, end_idx), desc="Processing rows: "):
        previous_generated = test_set.at[idx, "Generated_0"]  # Accessing Generated_0

        for i in range(1, len(test_set.columns) // 2):
            sentence_col = f"Sentence_{i}"
            generated_col = f"Generated_{i}"

            if sentence_col in test_set.columns and str(test_set.at[idx, sentence_col]) != "nan" and str(test_set.at[idx, sentence_col]) != "":
                current_sentence = test_set.at[idx, sentence_col]
                input_data = make_input_with_context(current_sentence, previous_generated)
                generated_sentence = _generate_with_context(input_data, **generate_config)
                
                if len(generated_sentence) > 0:  # if not blank
                    previous_generated = generated_sentence
                    test_set.at[idx, generated_col] = generated_sentence
            else:
                break

        # After each row, save the results to the CSV file
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
        generate_first_custom(test_set, config["BATCH_SIZE"], config["generated_path"], config["first_sentences_generation_config"])
        generate_first_custom_with_context(config["generated_path"], config["middle_sentences_generation_config"])
        if config["evaluate"]:
            evaluate_script.main(config["generated_path"], config["generated_metrics_path"])

    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)    

# if __name__ == '__main__':
#     main()

