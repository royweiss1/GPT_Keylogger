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
