import argparse
import sys
import os
import generate

def parse_arguments():
    parser = argparse.ArgumentParser(description="Welcome to the GPT Keylogger Framework. We offer three modes: Train, Generate and Plug & Play. Use the latter for trying the framework out.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', metavar='<configuration file>', type=str, help="Train the framework, with a configuration file.")
    group.add_argument('--generate', metavar='<configuration file>', type=str, help="Generate mode is used to evaluate the framework, with a configuration file.")
    group.add_argument('--play', action='store_true', help="Plug and Play mode, use the framework for specific samples. No configuration file is needed.")
    
    args = parser.parse_args()
    
    if (args.train and args.play) or (args.generate and args.play):
        parser.error("The arguments --train, --generate and --play are mutually exclusive.")
    
    return args

def validate_file_path(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    if args.train:
        validate_file_path(args.train)
        print(f"Training with configuration file: {args.train}")
        # Add your training code here
    elif args.generate:
        validate_file_path(args.generate)
        generate.main(args.generate)
        

    elif args.play:
        print("Playing mode activated.")
        # Add your play mode code here







if __name__ == '__main__':
    main()

