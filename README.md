# GPT Keylogger
This is the official repository for the code used in the paper: "What Was Your Prompt? A Remote Keylogging Attack on AI Assistants", USENIX Security 24'
# Description
We unveil a novel side-channel that can be used to read encrypted responses from AI Assistants over the web: the token-length side-channel. 
The attack relays on the fact that LLM Services sends their responses to the users in a streaming like manner. If each token (akin to word) is sent carelessly in a new packet, a MitM adversary that listens to the generated network can infer the size of the tokens of the AI Assistent response.
With this information in hand we show how a model can be trained to infer the topic of the asked response.

# Usage
