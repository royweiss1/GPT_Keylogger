import pandas as pd
import Levenshtein
import evaluate
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator
import torch
from torch import nn
import pickle
import csv

def format_floats(float_dict):
    return {key: f"{value:.4f}" for key, value in float_dict.items()}


def compute_metrics(model_sentence_transformers, reference_sentence, sentence_to_compare, evaluate_all_metrics):

    embed_pred = model_sentence_transformers.encode([sentence_to_compare], convert_to_tensor=True)
    embed_reference = model_sentence_transformers.encode([reference_sentence], convert_to_tensor=True)

    # Compute cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sen_trans_score = cos(embed_pred, embed_reference)
    cosine_score = tuple(sen_trans_score.detach().cpu().numpy())
    if not evaluate_all_metrics:
        return format_floats([cosine_score[0]])
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=[sentence_to_compare], references=[reference_sentence], use_aggregator=False)
    rouge_L = rouge_results["rougeL"][0]
    rouge_1 = rouge_results["rouge1"][0]
    
    ref = reference_sentence
    comp = sentence_to_compare
    length = max(len(ref), len(comp))
    ed = (length - Levenshtein.distance(ref, comp)) / length  
    ref_set, comp_set = set(ref.split()), set(comp.split())
    jac = float(len(ref_set & comp_set)) / len(ref_set | comp_set)
        
    return format_floats({"Cosine":cosine_score[0], "Rouge1":rouge_1, "RougeL":rouge_L, "Edit Distance":ed, "Jaccard":jac})


def calculate_scores(csv_file_path, output_csv_path, evaluate_all_metrics, start_idx=0, end_idx=None, checkpoint_interval=500):
    # Load the combined CSV file
    accelerator = Accelerator(cpu=False)
    torch.cuda.empty_cache()
    print("-------Device:", accelerator.device, "-------")
    model_sentence_transformers = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
    model_sentence_transformers = model_sentence_transformers.to(accelerator.device) 


    df = pd.read_csv(csv_file_path)

    # Determine the end index if not provided
    if end_idx is None:
        end_idx = len(df)

    # Initialize a list to hold all other scores
    csv_rows = []


    for paragraph_idx in tqdm(range(start_idx, end_idx), desc="Processing rows: "):
        
        reference_para = []
        generated_para = []
        
        for sentence_idx in range(45): # should be max 45 sentences
            sentence_col = f"Sentence_{sentence_idx}"
            generated_col = f"Generated_{sentence_idx}"

            if sentence_col in df.columns and generated_col in df.columns:
                reference_sentence = df.at[paragraph_idx, sentence_col]
                generated_sentence = df.at[paragraph_idx, generated_col]

                if str(reference_sentence) != "nan" and str(generated_sentence) != "nan":
                    reference_para.append(reference_sentence)
                    generated_para.append(generated_sentence)

                    ref_combined = "".join(reference_para)
                    gen_combined = "".join(generated_para)

                    scores = compute_metrics(model_sentence_transformers, ref_combined, gen_combined, evaluate_all_metrics)
                    row = {
                        "Paragraph": paragraph_idx,
                        "Sentence": sentence_idx,
                        **scores
                    }
                    csv_rows.append(row)


        # Save the checkpoint
        if (paragraph_idx + 1) % checkpoint_interval == 0:
            fieldnames = csv_rows[0].keys()
            with open(output_csv_path, 'w', newline='') as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)


    fieldnames = csv_rows[0].keys()
    with open(output_csv_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Results saved to {output_csv_path}")



# here you can run statistical analysis on the scores
def read_and_print_csv(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # take those who Sentence=0
    firsts = df[df["Sentence"] == 0]["Cosine"].values

    print(f"Attack Success Rate, first Sentences: {sum(1 for number in firsts if number[0] > 0.5) / len(firsts) * 100:.2f}%")
    print(f"Precentege of almost Identical Decyphering, first Sentences: {sum(1 for number in firsts if number[0] > 0.9) / len(firsts) * 100:.2f}%")

# read_and_print_pickle("scores.pkl")

def main(generated_output_path: str, generated_metrics_path: str, evaluate_all_metrics: bool):
    calculate_scores(generated_output_path, generated_metrics_path, evaluate_all_metrics)
    read_and_print_csv(generated_metrics_path)


