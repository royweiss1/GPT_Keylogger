# GPT Keylogger
In this repository you will find a python implementation of GPT Keylogger, a tool used for decyphering encrypted LLM's responses over the web.

This is the official repository for the code used in the paper: *["What Was Your Prompt? A Remote Keylogging Attack on AI Assistants"](https://arxiv.org/abs/2403.09751), USENIX Security 24'

*Weiss R, Ayzenshteyn D, Amit G, Mirsky Y "What Was Your Prompt? A Remote Keylogging Attack on AI Assistants", USNIX Security 24*
If you use any derivative of this code in your work, please cite our publicaiton. You may use the TODO FILL AT THE END BIBTEX

For more details you view our paper: *["here"](https://arxiv.org/abs/2403.09751)

# Description
Many modern LLM (AI assitant) services are vulnerable to a side channel attack that enables eavesdropping adversaries to read responses from the **service even though they are encrytped**...

![image](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/9c9f1bce-1bf2-4f02-902d-47249bf48a9c)

We unveil a novel side-channel that can be used to read encrypted responses from AI Assistants over the web: the token-length side-channel. 
The attack relays on the fact that LLM Services sends their responses to the users in a streaming like manner. If each token (akin to word) is sent carelessly in a new packet, a MitM adversary that listens to the generated network can infer the size of the tokens of the AI Assistent response.
With this information in hand we show how a model can be trained to infer the topic of the asked response.

## Contents


## Setup
  $ pip install https://github.com/royweiss1/GPT_Keylogger/archive/master.zip
  $ cd GPT_Keylogger
  $ pip install -r requirments.txt

## Usage
In this repository we offer a CLI tool capabale of:
1) Plug and Play - Using the framework to decypher a stream of token sizes extracted from a evesdropping the network
2) Generation & Evaluation - Using the framework on a spesified dataset, and evaluating it's preformance
3) Train - Training the framework from scrach (pre-trained) or from our proposed models (fine-tune) on a spesified dataset.
And We also offer python scripts and Jupyter notebook for each usecase for you convinience.






### Limitations
* Our models were trained on GPT-4 responses - Note that using it on a different LLM might cause worse results (due to different tokenizer and diffrent pattern of responses)
* Our model was trained using the UltraChat dataset. Using it on different datasets that includes different topics might lead to lower results.




### Plug and Play ###
We offer a plug and play Jupyter Notebook - Demonstration.ipynb. There you can simply enter the packet sizes from the pcap file. Then the model will generate to you the responses sorted by the model's confidence score.

### Generation ###
In order to generate the responses on a full scale you can use generate.py. There a user can generate full paragraphs using the models we have fine-tuned (which are hosted on Huggingface).

### Evaluate ###
We have also included a jupyter notebook with all of the relevant metrics in order to fully evaluate the generated paragraphs/sentneces. The notebook includes both text analysis metrics such as Rouge and Edit Distance and Topic analysis using Sentence Transformer's embeddings cosine simularity. 
