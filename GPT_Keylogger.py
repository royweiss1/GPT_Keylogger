print("GPT Keylogger Framework")
print("Loading...")

import argparse
import sys
import os
import generate
import plug_play

def parse_arguments():
    parser = argparse.ArgumentParser(description="Welcome to the GPT Keylogger Framework. We offer several modes: Train, Generate, and Plug & Play. Use the latter for trying the framework out.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train-first-sentences', '-tf', metavar='<configuration file>', type=str, help="Train the framework's first sentences model, with a configuration file.")
    group.add_argument('--train-middle-sentences', '-tm', metavar='<configuration file>', type=str, help="Train the framework's first sentences model, with a configuration file.")
    group.add_argument('--generate', '-g', metavar='<configuration file>', type=str, help="Generate mode is used to evaluate the framework, with a configuration file.")
    group.add_argument('--play', '-p', action='store_true', help="Plug and Play mode, use the framework for specific samples. No configuration file is needed.")
    
    args = parser.parse_args()
    
    if sum([args.train_first_sentences is not None, args.train_middle_sentences is not None, args.generate is not None, args.play]) > 1:
        parser.error("The arguments --train-first-sentences, --train-middle-sentences, --generate, and --play are mutually exclusive.")
    
    return args


def validate_file_path(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    if args.train_first_sentences:
        validate_file_path(args.train)
        print(f"Training with configuration file: {args.train}")
        # Add your training code here
        
    elif args.train_middle_sentences:
        validate_file_path(args.train)
        print(f"Training with configuration file: {args.train}")
        # Add your training code here

    elif args.generate:
        validate_file_path(args.generate)
        generate.main(args.generate)
        

    elif args.play:
        plug_play.main()






if __name__ == '__main__':
    main()

