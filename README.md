# An NTM for language tasks
## Overview:
This is a project where I seek to understand the transformer architecture in depth annd modify it with another architecture (NTM, Graves et al., 2016) to hopefully produce some interesting results in search problems prompted with language. 
## Project Goals:
- [ ] Take notes throughout the dev process
- [ ] Understand Attention Mechanisms in seq2seq models
- [ ] Implement a basic transformer from Attention Is All You Need (2017)
- [ ] Train using some basic dataset
- [ ] Complete gaps in knowledge (marked by hows)
- [ ] Implement an NTM
- [ ] Connect the two
- [ ] Do some kool experiments
- [ ] Upgrade to DNC
## A Transformer:
### Background:
Before transformers came seq2seq models implemented with RNNs. These also used encoder decoder models where the encoder would read a word at a time from the input, encode all this into some representation and produce output tokens conditioned on this representation. In this way, the decoder only receives a sort of summarized input. The attention mechanism was introduced to help the decoder "attend" to various positions of input encodings (the word alignment is used to determine what input tokens output tokens "align" to, a consequence of attention mechanisms).<br>
After all the encoder steps are done, the decoder pays attention to both encoder hidden states and summaries giving the model "context" to take into account.
### Attention
The attention mechanism takes three entities into account, queries (Q), keys (K), and values (V). Q corresponds to an encoding from the decoder's hidden state, whereas K and V are from the encoders hidden state. The idea is to produce a weighting (using softmax) to decide what part of the encoder's hidden state (inputs) the decoders next output should focus on (attention lol). <br>
<p align="center">
  <img src="https://github.com/mishaalkandapath/neuralTuringMachine/blob/main/notes/attn.png" alt="attention" width=50% />
</p><br>
[source](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)<br>
This weighted average is called the <b> context vector </b>. <br>
In the attention is al you need paper, compatibility is computed through a dot product. In predecessor seq2seq models, such a weighted attention was not presented, instead the encoding is only based on the current input hidden state. <br>

### Self Attention
Apart from the abovementioned encoder-decoder attention, self attention is an additional attention method in Attention is All You Need. It gets Q, K, and V all from the encoder. Q is obtained as a transformation on the encoder hidden state, similarly (but a different transformation ofc) from K and V.
<p align="center">
  <img src="https://github.com/mishaalkandapath/neuralTuringMachine/blob/main/notes/selfattn.png" alt="self_attention" width=50% />
</p>
<br>
Note that the boxes correspond to the dimension of the corresponding vectors. The weights to learn these tranformations have to be learnt (say from a cost function of the produced outputs). Read more at [source](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a) (also the source of the above image).

<br>
We need a transformation as otherwise, the weight of the vector with itself will be higher. So in calculating attention at an input token, if the current query is of that input, then all/most of the attention will be on that input itself other than being distributed across other input tokens.

### The difference:
The first attention allows for understanding context from the input in producing the output, thus the compatibility is between the encoder and decoder encodings. Self-attention helps new input encodings out by determining what in the preceding sequence it should take into account while producing an input encoding. Thus, the compatibility is between two positions in the input sequence (the current and some of the earlier ones). 

### Why self attention?
Three major points were listed by the paper, of which the first two are total computational complexity per layer and the other is increased parallizability. <br>
The third and major one is the "path length between long-range dependencies between the network", i.e, the length signals (forward and backward) have to travel as a function of positions in the input and output is less than preceding methods. Thus, it is easier to learn long range dependencies due to this shorter traversal length. <br>
As a plus, it also allows for more interpretability.
