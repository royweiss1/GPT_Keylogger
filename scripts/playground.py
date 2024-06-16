from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator
import torch
import tiktoken
import regex as re


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



def encode_text(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_text = enc.encode(text)
    return tokenized_text


def decode_text(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    decoded_tokens = [enc.decode([token]) for token in text]
    return decoded_tokens


def apply_token(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_text = [enc.decode([token]) for token in enc.encode(text)]
    return tokenized_text



def concatenate_sentences(answer):
    new_answer = []
    index = 0
    while index < len(answer) - 1:
        sentence = answer[index]
        if re.search(r':\n\n\d\.', sentence):
            if len(sentence) <= 90:
                new_answer.append(sentence + answer[index + 1])
                index += 1
            else:
                new_answer.append(sentence)
        else:
            new_answer.append(sentence)
        index += 1
    return new_answer


def heuristic_string(answer: str):
    sentences = []
    decoded = decode_text(encode_text(answer))
    lengths = [len(dec) for dec in decoded]
    index = 0
    tokens_in_streak = 0
    while index < len(lengths):
        if tokens_in_streak >= 10 and lengths[index] == 1:
            if lengths[index-1] == 3:
                sentences.append("".join(decoded[:tokens_in_streak]))
                decoded = decoded[tokens_in_streak:]
            elif lengths[index-1] == 1:
                sentences.append("".join(decoded[:tokens_in_streak-1]))
                decoded = decoded[tokens_in_streak-1:]
                index -= 1
            else:
                sentences.append("".join(decoded[:tokens_in_streak+1]))
                decoded = decoded[tokens_in_streak+1:]
                index += 1
            tokens_in_streak = 0
        else:
            index += 1
            tokens_in_streak += 1
    else:
        if tokens_in_streak > 0:
            sentences.append("".join(decoded))
    answer = concatenate_sentences(answer)
    return sentences
        

def heuristic_list(lengths: list):
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
    type_of_input = input("What do you wish to enter? Test Paragraph (1) or Token Sizes (2) or Packet Lengths (3): ")
    while type_of_input != "1" and type_of_input != "2" and type_of_input != "3":
        type_of_input = input("What do you wish to enter? Test Paragraph (1) or Token Sizes (2) or Packet Lengths (3): ")
    
    input_str = ""
    while input_str.lower() != "yes":
        if type_of_input == "1":
            input_str = input("Enter the paragraph or first Sentence: ")
            token_lens = heuristic_string(input_str)[0] # take the first sentence based on the heuristic
        else:
            input_str = input("Enter the lengths divided by comma (,): ")
            token_lens = [int(x) for x in input_str.split(",")]
            if type_of_input == "3":
                token_lens = parse_packet_lengths(token_lens)
                token_lens = heuristic_list(token_lens)[0] # take the first sentence based on the heuristic
        
        print("Token lengths:", token_lens)
        print("Generating first sentences...")
        outputs = generate_first([make_input(token_lens)]) # output is sorted by the model's confidence!!!
        print("First sentences generated, ranked by the model's confidence:")
        for rank, output in enumerate(outputs):
            print(f"Rank: {rank+1}. Output: {output}")
        print("*"*50)
        input_str = input("Do you wish to stop? yes/no: ")
