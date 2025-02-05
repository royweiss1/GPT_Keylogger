<a name="readme-top"></a>

# GPT Keylogger
In this repository, you will find a Python implementation of a tool for deciphering encrypted responses sent from AI assistants (such as ChatGPT, Copilot, ...).

This is the official repository for the code used in the paper:
["What Was Your Prompt? A Remote Keylogging Attack on AI Assistants"](https://www.usenix.org/conference/usenixsecurity24/presentation/weiss), USENIX Security 24'

*Weiss R, Ayzenshteyn D, Amit G, Mirsky Y "What Was Your Prompt? A Remote Keylogging Attack on AI Assistants", USNIX Security 24*

If you use any derivative of this code in your work, please cite our publication. You can find a BibTex citation below.


# Description
Many modern LLM (AI assistant) services are vulnerable to a side-channel attack that enables eavesdropping adversaries to read responses from the **service even though they are encrypted**. Imagine awkwardly asking your AI about a strange rash, or to edit an email, only to have that conversation exposed to someone on the net. In this repository we demonstrate how our unveiled side channel can be used to read encrypted messages sent from AI Assistants. The figure below shows the general idea of the Side Channel:

![image](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/9c9f1bce-1bf2-4f02-902d-47249bf48a9c)

We unveil a novel token-length side-channel. The attack relies on the fact that LLM Services sends their responses to the users in a streaming-like manner. If each token (akin to a word) is sent carelessly in a new packet, a MitM adversary that listens to the generated network can infer the size of the tokens of the AI Assistent response. With this information in hand, we show how a model can be trained to infer the topic of the asked response.

## Features
* Our models were trained on GPT-4 responses - note that using it on different LLMs might cause worse results (due to different tokenizers and different patterns of responses)
* Our model was trained using the UltraChat dataset. Using it on different datasets that includes different topics might lead to lower results.

💡 **Note: The tool in this repository does not operate on network captures (e.g., PCAPs). You must extract and provide the token sequences yourself**

## Contents
1) Setup - Simple installation of required libraries
2) General Usage:
   * Playground Module
   * Generate & Evaluate Module
   * Train Module
3) Examples - From pcap files + A demo video

## Setup
```
git clone https://github.com/royweiss1/GPT_Keylogger.git
cd GPT_Keylogger
pip install -r requirements.txt
```


## Usage
In this repository, we offer a CLI tool capable of:
1) Playground - Using the framework to decypher a stream of token sizes extracted from an eavesdropping network.
2) Generation & Evaluation - Using the framework on a specified dataset, and evaluating its performance
3) Train - Training the framework from Scratch (pre-trained) or from our proposed models (fine-tune) on a specified dataset.

For all of the use cases above we also offer Python scripts and Jupyter notebooks for your convenience.

```
Welcome to the GPT Keylogger Framework. We offer several modes: Train, Generate, and Playground. Use the latter to try the
framework out.

options:
  -h, --help            show this help message and exit
  --train-first-sentences <configuration file>, -tf <configuration file>
                        Train the framework's first sentences model, with a configuration file.
  --train-middle-sentences <configuration file>, -tm <configuration file>
                        Train the framework's first sentences model, with a configuration file.
  --generate <configuration file>, -g <configuration file>
                        Generate mode is used to evaluate the framework, with a configuration file.
  --Playground, -p            Playground mode, use the framework for specific samples. No configuration file is needed.
```

### Playground ###
This module is the simplest module. Use it to try out the framework. It can receive two kinds of input:
1. A sequence of **token** lengths
2. A sequence of **packet** lengths
The difference is that packet lengths are taken straight up from the pcap file. They are incremental, meaning the response is building up over the stream of packets. For example, the packets may contain:
```
I, I can, I can't, I can't diagnose, I can't diagnose medical .....
```
Use it with:
```
python GPT_Keylogger.py --Playground
```

This module can also be found in the Playground.ipynb notebook.

### Generation ###
This module offers a full evaluation of our framework against an arbitrary AI Assistant responses dataset. We offer the option to both generate deciphered responses and also evaluate their similarity to the original responses.

Note: Use this module if you wish to reproduce our results from the paper, using the default configuration file.

To use it you need to specify a generation-configuration file (an example could be found under `config/generation_config.json`). In the config file there are specifications for:
- What path should the results be saved to.
- What model should be used.
- What hyperparameters should be used to generated the responses using the models.

We offer a default configuration file for your ease of use.

The only requirement is that the dataset must follow the following format:
```
{
    "paragraphs": [
        [
          "Sentence1",
          "Sentence2",
          ...
        ],
        [
          ...
        ]
    ]
}
```
The partition of sentences may be done using a heuristic we propose (on `archive/hueristic.ipynb`).

We offer the test set that was used in our paper at: `data\test.json`. It is also available at Huggingface with the rest of the dataset: [here](https://huggingface.co/datasets/royweiss1/GPT_Keylogger_Dataset)

Make sure to run:
```
python GPT_Keylogger.py --generate config/generation_config.json
```

### Train ###
This module is composed of 2 modules. As described in our paper our framework is composed of 2 T5 models which have been fine-tuned for different missions. We took the desired paragraph and divided it into sentences. The first sentence was passed into the model which was fine-tuned on the first sentences. The i_th (i>1) was passed into the second model which was trained on 'middle' sentences. The i_th sentence was given as input with the (i-1)_th sentence as context. This way we have built an entire paragraph.

In this module, we offer the user the to train a model for this task. The user must supply a configuration file, which can be found at `config/training_config.json`.

For training the first sentences:

☝🏻 **Make sure you have set up the train.json and validation.json in the correct path specified in the configuration file.**

```
python GPT_Keylogger.py --train-first-sentences config/training_config.json
python GPT_Keylogger.py -tf config/training_config.json
```
```
python GPT_Keylogger.py --train-middle-sentences config/training_config.json
python GPT_Keylogger.py -tm config/training_config.json
```

## Examples
Here are examples of reconstructions of Responses that were extracted from real PCAP files. We eavesdropped real conversations with ChatGPT and Microsoft Copilot.

![Screenshot 2024-05-26 at 15 11 07](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/31e34335-7c52-435b-83e8-30669785c06c)
![Screenshot 2024-05-26 at 15 11 16](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/cc2002aa-9f05-4957-bee5-81e40fb68c49)

We also provide a YouTube demonstration video of the attack: [here](https://www.youtube.com/watch?v=UfenH7xKO1s&t)


https://github.com/royweiss1/GPT_Keylogger/assets/92648019/0abee8e1-a04f-42a3-b4ed-d2e1dda78caa



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

### Citation

```
@inproceedings {weiss2024your,
author = {Roy Weiss and Daniel Ayzenshteyn and Guy Amit and Yisroel Mirsky},
title = {What Was Your Prompt? A Remote Keylogging Attack on {AI} Assistants},
booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
year = {2024},
isbn = {978-1-939133-44-1},
address = {Philadelphia, PA},
pages = {3367--3384},
url = {https://www.usenix.org/conference/usenixsecurity24/presentation/weiss},
publisher = {USENIX Association},
month = aug
}
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

