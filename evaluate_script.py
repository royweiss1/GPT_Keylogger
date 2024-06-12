import pandas as pd
import Levenshtein
import evaluate_script
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator
import torch
import numpy as np
from torch import nn
import pickle


def format_floats(float_list):
    return [float(f"{num:.3f}") for num in float_list]


def compute_metrics(reference_sentence, sentence_to_compare):
    accelerator = Accelerator(cpu=False)
    torch.cuda.empty_cache()
    print("-------Device:", accelerator.device, "-------")
    model_sentence_transformers = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
    model_sentence_transformers = model_sentence_transformers.to(accelerator.device) 

    embed_pred = model_sentence_transformers.encode([sentence_to_compare], convert_to_tensor=True)
    embed_reference = model_sentence_transformers.encode([reference_sentence], convert_to_tensor=True)

    # Compute cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sen_trans_score = cos(embed_pred, embed_reference)
    cosine_score = tuple(sen_trans_score.detach().cpu().numpy())

    rouge = evaluate_script.load('rouge')
    rouge_results = rouge.compute(predictions=[sentence_to_compare], references=[reference_sentence], use_aggregator=False)
    rouge_L = rouge_results["rougeL"][0]
    rouge_1 = rouge_results["rouge1"][0]
    
    ref = reference_sentence
    comp = sentence_to_compare
    length = max(len(ref), len(comp))
    ed = (length - Levenshtein.distance(ref, comp)) / length  
    ref_set, comp_set = set(ref.split()), set(comp.split())
    jac = float(len(ref_set & comp_set)) / len(ref_set | comp_set)
        
    return format_floats([cosine_score, rouge_1, rouge_L, ed, jac])


def calculate_scores(csv_file_path, output_pickle_path, start_idx=0, end_idx=None, checkpoint_interval=500):
    # Load the combined CSV file
    df = pd.read_csv(csv_file_path)

    # Determine the end index if not provided
    if end_idx is None:
        end_idx = len(df)

    # Initialize a list to hold all other scores
    all_other_scores = []

    for idx in tqdm(range(start_idx, end_idx), desc="Processing rows: "):
        row_other_scores = []
        
        reference_para = []
        generated_para = []
        
        for i in range(45): # should be max 45 sentences
            sentence_col = f"Sentence_{i}"
            generated_col = f"Generated_{i}"

            if sentence_col in df.columns and generated_col in df.columns:
                reference_sentence = df.at[idx, sentence_col]
                generated_sentence = df.at[idx, generated_col]

                if str(reference_sentence) != "nan" and str(generated_sentence) != "nan":
                    reference_para.append(reference_sentence)
                    generated_para.append(generated_sentence)

                    ref_combined = "".join(reference_para)
                    gen_combined = "".join(generated_para)

                    scores = compute_metrics(ref_combined, gen_combined)
                    row_other_scores.append(scores)
        
        all_other_scores.append(row_other_scores)

        # Save the checkpoint
        if (idx + 1) % checkpoint_interval == 0:
            with open(output_pickle_path, 'wb') as checkpoint_file:
                pickle.dump(all_other_scores, checkpoint_file)

    # Save the final results to the pickle file
    with open(output_pickle_path, 'wb') as output_file:
        pickle.dump(all_other_scores, output_file)

    print(f"Results saved to {output_pickle_path}")



# here you can run statistical analysis on the scores
def read_and_print_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Print the contents of the pickle file
    firsts = [row[0] for row in data]

    print(f"Attack Success Rate, first Sentences: {sum(1 for number in firsts if number > 0.5) / len(firsts) * 100:.2f}%")
    print(f"Precentege of almost Identical Decyphering, first Sentences: {sum(1 for number in firsts if number > 0.9) / len(firsts) * 100:.2f}%")

# read_and_print_pickle("scores.pkl")

def main(generated_output_path: str, generated_metrics_path: str):
    calculate_scores(generated_output_path, generated_metrics_path)
    read_and_print_pickle(generated_metrics_path)


