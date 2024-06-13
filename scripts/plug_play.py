from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator
import torch


def generate_first(encodings):
    print("Loading Model and Tokenizer to device...")

    MAX_LENGTH = 80
    NUM_OF_OUTPUTS = 32

    first_model_checkpoint = "royweiss1/T5_FirstSentences"
    first_model = AutoModelForSeq2SeqLM.from_pretrained(first_model_checkpoint)
    first_tokenizer = AutoTokenizer.from_pretrained(first_model_checkpoint)

    torch.cuda.empty_cache()
    accelerator = Accelerator(cpu=False)
    print("-------Device:", accelerator.device)
    first_model = first_model.to(accelerator.device)
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
    
    sorted_indices = sequence_scores.argsort(descending=True)

    sorted_sequences = sequences[sorted_indices]

    sorted_texts = [first_tokenizer.decode(seq, skip_special_tokens=True) for seq in sorted_sequences]

    return sorted_texts


def parse_packet_lengths(lengths):
    token_lengths = []
    for i in range(1, len(lengths)):
        token_lengths.append((lengths[i] - lengths[i-1])//2)
    return token_lengths
        

def heuristic(lengths):
    sentences = []
    index = 0
    tokens_in_streak = 0
    while index < len(lengths):
        if tokens_in_streak >= 10 and lengths[index] == 1:
            if lengths[index-1] == 3:
                sentences.append(lengths[:tokens_in_streak])
                lengths = lengths[tokens_in_streak:]
            elif lengths[index-1] == 1:
                sentences.append(lengths[:tokens_in_streak-1])
                lengths = lengths[tokens_in_streak-1:]
                index -= 1
            else:
                sentences.append(lengths[:tokens_in_streak+1])
                lengths = lengths[tokens_in_streak+1:]
                index += 1
            tokens_in_streak = 0
        else:
            index += 1
            tokens_in_streak += 1
    else:
        if tokens_in_streak > 0:
            sentences.append(lengths)
    return sentences


def make_input(lst_lengths):
    lst_str = " ".join([f"_{i}" for i in lst_lengths])
    return f"Translate the Special Tokens to English. \nSpecial Tokens:{lst_str}"


def main():
    packet_lens = False
    input_str = input("Please enter the correct number based of the kind of input that you wish to enter: Token Sizes (1) or Packet Lengths (2): ")
    while input_str != "1" and input_str != "2":
        input_str = input("Please enter the correct number based of the kind of input that you wish to enter: Token Sizes (1) or Packet Lengths (2): ")
    if input_str == "2":
        packet_lens = True

    input_str = input("Enter the lengths of the packets divided by comma (,) or -1 if you wish to stop: ")
    while input_str != "-1":
        token_lens = [int(x) for x in input_str.split(",")]
        if packet_lens:
            token_lens = parse_packet_lengths(token_lens)
        print("Token lengths:", token_lens)
        print("Generating first sentences...")
        token_lens = heuristic(token_lens)[0] # take the first sentence based on the heuristic
        outputs = generate_first([make_input(token_lens)]) # output is sorted by the model's confidence!!!
        print("First sentences generated, ranked by the model's confidence:")
        for rank, output in enumerate(outputs):
            print(f"Rank: {rank+1}. Output: {output}")
        print("*"*50)
        input_str = input("Enter the lengths of the packets divided by comma (,) or -1 if you wish to stop: ")