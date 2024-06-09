# GPT Keylogger
This is the official repository for the code used in the paper: "What Was Your Prompt? A Remote Keylogging Attack on AI Assistants", USENIX Security 24'
https://arxiv.org/abs/2403.09751
# Description
We unveil a novel side-channel that can be used to read encrypted responses from AI Assistants over the web: the token-length side-channel. 
The attack relays on the fact that LLM Services sends their responses to the users in a streaming like manner. If each token (akin to word) is sent carelessly in a new packet, a MitM adversary that listens to the generated network can infer the size of the tokens of the AI Assistent response.
With this information in hand we show how a model can be trained to infer the topic of the asked response.

# Usage
### Plug and Play ###
We offer a plug and play Jupyter Notebook - Demonstration.ipynb. There you can simply enter the packet sizes from the pcap file. Then the model will generate to you the responses sorted by the model's confidence score.

### Generation ###
In order to generate the responses on a full scale you can use generate.py. There a user can generate full paragraphs using the models we have fine-tuned (which are hosted on Huggingface).

### Evaluate ###
We have also included a jupyter notebook with all of the relevant metrics in order to fully evaluate the generated paragraphs/sentneces. The notebook includes both text analysis metrics such as Rouge and Edit Distance and Topic analysis using Sentence Transformer's embeddings cosine simularity. 
