<a name="readme-top"></a>

# GPT Keylogger
In this repository you will find a python implementation of a tool for deciphering encypeted responses sent from AI assistants (such as ChatGPT, Copilot, ...).

This is the official repository for the code used in the paper:
["What Was Your Prompt? A Remote Keylogging Attack on AI Assistants"](https://arxiv.org/abs/2403.09751), USENIX Security 24'

*Weiss R, Ayzenshteyn D, Amit G, Mirsky Y "What Was Your Prompt? A Remote Keylogging Attack on AI Assistants", USNIX Security 24*

If you use any derivative of this code in your work, please cite our publicaiton. You can find a BibTex citation below.


# Description
Many modern LLM (AI assitant) services are vulnerable to a side channel attack that enables eavesdropping adversaries to read responses from the **service even though they are encrytped**. Imagine awkwardly asking your AI about a strange rash, or to edit an email, only to have that conversation exposed to someone on the net. In this repository we demonstrate how our unvailed side channel can be used to read encrypted messages sent from AI Assistants. The figure below shows the general idea of the Side Channel:

![image](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/9c9f1bce-1bf2-4f02-902d-47249bf48a9c)

We unveil a novel token-length side-channel. The attack relays on the fact that LLM Services sends their responses to the users in a streaming like manner. If each token (akin to word) is sent carelessly in a new packet, a MitM adversary that listens to the generated network can infer the size of the tokens of the AI Assistent response. With this information in hand we show how a model can be trained to infer the topic of the asked response.

## Features
* Our models were trained on GPT-4 responses - Note that using it on a different LLM might cause worse results (due to different tokenizer and diffrent pattern of responses)
* Our model was trained using the UltraChat dataset. Using it on different datasets that includes different topics might lead to lower results.

💡 **Note: The tool in this repository does not operate on network captures (e.g., pcaps). You must extract and provide the token sequences yourself**

## Contents
1) Setup - Simple installation of required libraries
2) General Usage:
   * Playground Module
   * Generate & Evaluate Module
   * Train Module
3) Examples - From pcap files + A demo video

## Setup
```
pip install https://github.com/royweiss1/GPT_Keylogger.git
cd GPT_Keylogger
pip install -r requirments.txt
```


## Usage
In this repository we offer a CLI tool capabale of:
1) Playground - Using the framework to decypher a stream of token sizes extracted from a evesdropping the network.
2) Generation & Evaluation - Using the framework on a spesified dataset, and evaluating it's preformance
3) Train - Training the framework from scrach (pre-trained) or from our proposed models (fine-tune) on a spesified dataset.

For all of the usecases above we also offer python scripts and Jupyter notebooks for you convinience.

```
Welcome to the GPT Keylogger Framework. We offer several modes: Train, Generate, and Playground. Use the latter for trying the
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
This module is the simplest module. Use it to try out the framework. It can recive two kinds of input:
1. A sequence of **token** lengths
2. A sequence of **packet** lengths
The difference is that packet lengths are taken straight up from the pcap file. They are incremental, meaning the response is beeing build up over the stream of packets. For example, the packets may contain:
```
I, I can, I can't, I can't diagnose, I can't diagnose medical .....
```
Use it with:
```
python GPT_Keylogger.py --Playground
```

This module can also be found in the Playground.ipynb notebook.

### Generation ###
This module offers full evaluation of our framework against an arbitrary AI Assistent responses dataset. We offer the option to both generate decyphered responses and also evaluate their similarity to the original responses.

Note: Use this module if you wish to repreduce our results from the paper, using the default configuration file.

To use it you need to spesify a generation-configuration file (an example could be found under `config/generation_config.json`). In the config file there are spesifications on:
- What path should the results be saved to.
- What model should be used.
- What hyperparameters should be used to generated the responses using the models.

We offer a default configuration file for you ease of use.

The only requerment is that the dataset that is used will follow the following format:
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
The partition to sentences may be done using a hueristic we propose (on `archive/hueristic.ipynb`).

We offer the test set that was used in our paper at: `data\test.json`. It also avaliable at huggingface with the rest of the dataset: [here](https://huggingface.co/datasets/royweiss1/GPT_Keylogger_Dataset)

Make sure to run:
```
python GPT_Keylogger.py --generate config/generation_config.json
```

### Train ###
This module is composed of 2 modules. As described in our paper our framework is compoed of 2 T5 models which have been fine tuned for different missions. We took the desired paragraph and divided it into sentences. The first sentences was passed into the model which was fined tuned on the first sentences. The i_th (i>1) was passed into the second model which was trained on 'middle' sentences. The i_th sentences was given as input with the (i-1)_th setences as context. This way we have build an entire paragraph.

In this module we offer the user the to train a model for this task. The user must supply a configuration file, which can be found at `config/training_config.json`.

For training the first sentences:

☝🏻 **Make Sure you have setup the train.json and validation.json in the correct path spesified at the configuration file.**

```
python GPT_Keylogger.py --train-first-sentences config/training_config.json
python GPT_Keylogger.py -tf config/training_config.json
```
```
python GPT_Keylogger.py --train-middle-sentences config/training_config.json
python GPT_Keylogger.py -tm config/training_config.json
```

## Examples
Here are examples of reconstructions of Responses which were extracted from real PCAP files. We eavesdropped real conversations with ChatGPT and Microsoft Copilot.

![Screenshot 2024-05-26 at 15 11 07](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/31e34335-7c52-435b-83e8-30669785c06c)
![Screenshot 2024-05-26 at 15 11 16](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/cc2002aa-9f05-4957-bee5-81e40fb68c49)

We Also provide a Youtube demonstration video of the attack: [here](https://www.youtube.com/watch?v=UfenH7xKO1s&t)


https://github.com/royweiss1/GPT_Keylogger/assets/92648019/0abee8e1-a04f-42a3-b4ed-d2e1dda78caa



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

### Citation

```
@inproceedings{weissLLMSideChannel,
  title={What Was Your Prompt? A Remote Keylogging Attack on AI Assistants},
  author={Weiss, Roy and Ayzenshteyn, Daniel and Amit Guy and Mirsky, Yisroel}
  booktitle={USENIX Security},
  year={2024}
}
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

