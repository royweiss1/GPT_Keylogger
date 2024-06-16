print("GPT Keylogger Framework")
print("Loading...")
import scripts.train as train

import argparse
import sys
import os
import scripts.generate as generate
import scripts.plug_play as plug_play

# silent warnings:
import warnings
warnings.filterwarnings("ignore", message="A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Welcome to the GPT Keylogger Framework. We offer several modes: Train, Generate, and Plug & Play. Use the latter for trying the framework out.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train-first-sentences', '-tf', metavar='<configuration file>', type=str, help="Train the framework's first sentences model, with a configuration file.")
    group.add_argument('--train-middle-sentences', '-tm', metavar='<configuration file>', type=str, help="Train the framework's middle sentences model, with a configuration file.")
    group.add_argument('--generate', '-g', metavar='<configuration file>', type=str, help="Generate mode is used to evaluate the framework, with a configuration file.")
    group.add_argument('--play', '-p', action='store_true', help="Plug and Play mode, use the framework for specific samples. No configuration file is needed.")
    
    return parser.parse_args()


def validate_file_path(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    if args.train_first_sentences:
        validate_file_path(args.train_first_sentences)
        train.main(args.train_first_sentences, first_sentences=True)
        
    elif args.train_middle_sentences:
        validate_file_path(args.train_middle_sentences)
        train.main(args.train_middle_sentences, first_sentences=False)
        
    elif args.generate:
        validate_file_path(args.generate)
        generate.main(args.generate)
        
    elif args.play:
        plug_play.main()


if __name__ == '__main__':
    main()

