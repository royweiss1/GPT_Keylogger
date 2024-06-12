<a name="readme-top"></a>

# GPT Keylogger
In this repository you will find a python implementation of GPT Keylogger, a tool used for decyphering encrypted LLM's responses over the web.

This is the official repository for the code used in the paper:
["What Was Your Prompt? A Remote Keylogging Attack on AI Assistants"](https://arxiv.org/abs/2403.09751), USENIX Security 24'

*Weiss R, Ayzenshteyn D, Amit G, Mirsky Y "What Was Your Prompt? A Remote Keylogging Attack on AI Assistants", USNIX Security 24*

If you use any derivative of this code in your work, please cite our publicaiton. You can find a BibTex citation below.

For more details you view our paper:
[here](https://arxiv.org/abs/2403.09751)

# Description
Many modern LLM (AI assitant) services are vulnerable to a side channel attack that enables eavesdropping adversaries to read responses from the **service even though they are encrytped**...

![image](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/9c9f1bce-1bf2-4f02-902d-47249bf48a9c)

We unveil a novel side-channel that can be used to read encrypted responses from AI Assistants over the web: the token-length side-channel. 
The attack relays on the fact that LLM Services sends their responses to the users in a streaming like manner. If each token (akin to word) is sent carelessly in a new packet, a MitM adversary that listens to the generated network can infer the size of the tokens of the AI Assistent response.
With this information in hand we show how a model can be trained to infer the topic of the asked response.

## Contents


## Setup
```
pip install https://github.com/royweiss1/GPT_Keylogger.git
cd GPT_Keylogger
pip install -r requirments.txt
```


## Usage
In this repository we offer a CLI tool capabale of:
1) Plug and Play - Using the framework to decypher a stream of token sizes extracted from a evesdropping the network.
2) Generation & Evaluation - Using the framework on a spesified dataset, and evaluating it's preformance
3) Train - Training the framework from scrach (pre-trained) or from our proposed models (fine-tune) on a spesified dataset.

For all of the usecases above we also offer python scripts and Jupyter notebooks for you convinience.

```

```

### Plug and Play ###
This module is the simplest module. Use it to try out the framework. It can recive two kinds of input:
1. A sequence of **token** lengths
2. A sequence of **packet** lengths
The difference is that packet lengths are taken straight up from the pcap file. They are incremental, meaning the response is beeing build up over the stream of packets. For example, the packets may contain:
```
I, I can, I can't, I can't diagnose, I can't diagnose medical .....
```
This module can also be found in the PlugAndPlay.ipynb notebook.

### Generation ###
In order to generate the responses on a full scale you can use generate.py. There a user can generate full paragraphs using the models we have fine-tuned (which are hosted on Huggingface).
We have also included a jupyter notebook with all of the relevant metrics in order to fully evaluate the generated paragraphs/sentneces. The notebook includes both text analysis metrics such as Rouge and Edit Distance and Topic analysis using Sentence Transformer's embeddings cosine simularity.


### Train ###


## Examples
Here are examples of reconstructions of Responses which were extracted from real PCAP files. We eavesdropped real conversations with ChatGPT and Microsoft Copilot.

![Screenshot 2024-05-26 at 15 11 07](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/31e34335-7c52-435b-83e8-30669785c06c)
![Screenshot 2024-05-26 at 15 11 16](https://github.com/royweiss1/GPT_Keylogger/assets/92648019/cc2002aa-9f05-4957-bee5-81e40fb68c49)

We Also provide a Youtube demonstration video of the attack: [here](https://www.youtube.com/watch?v=UfenH7xKO1s&t)


https://github.com/royweiss1/GPT_Keylogger/assets/92648019/0abee8e1-a04f-42a3-b4ed-d2e1dda78caa




### Limitations
* Our models were trained on GPT-4 responses - Note that using it on a different LLM might cause worse results (due to different tokenizer and diffrent pattern of responses)
* Our model was trained using the UltraChat dataset. Using it on different datasets that includes different topics might lead to lower results.


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

