# An NTM for language tasks
## Overview:
This is a project where I seek to understand the transformer architecture in depth. I also intend to build everything from scratch and in JAX.
## Project Goals:
- [ ] Take notes throughout the dev process
- [ ] Understand Attention Mechanisms in seq2seq models
- [ ] Implement a basic transformer from Attention Is All You Need (2017)
- [ ] Train using some basic dataset
- [ ] Complete gaps in knowledge (marked by hows)
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

### Multihead Attention
Instead of performing attention just once, the transformer model decides its better to compute h parallel attentions. To induce a fake richness in "perspective" you first transform the keys, values, and queries to dk, dv, dq dimensional spaces h times (differently). Then conduct self attention on all of them (so h perspectives). We then concatenate the information and convert it back into dmodel space using another transformation (linear). 

### Using Attention
1. encoder-decoder attention, where the queries are from the previous decoder layer, rest from encoder output.
2. self attention in the encoder, where each position in the encoder can attend to all positions in the previous encoding layer. queries, keys and values come from the previous layer
3. self attention in the decoder, each position in the decoder to attend to upto all positions including current. Information is to flow forwards only, and this is ensured by setting all illegal values in the value vector to -inf.

### Position-Wise Feed Forward NN
in the original paper, apart from the attention parts in a sublayer, they consist of a Position wise feed forward nn too given by:
<p align="center">
  $$ FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2}$$
</p>
<p>
  This is applied to every position the same way but with different weights. In the paper, $W_{1}$ is of dimensions $d_{model} \times d_{ff} = 512 \times 2048$ and $W_{2}$ is of the opposite dimension. 
</p>

### Input
Each vector represents the semantics and position of a token. The first step is to learn embeddings for each token and this is supplied into the encoders. Then the decoder produces one word at a time, which is fed back into the output embedder ot make the whole thing a chain. At the very beginning, there is nothing in the output embedder so a <SOS> token is provided. The decoder produces output probabilities of different words in vocabulary. Highes is chosen, fed into the output embedder and the process continues. 

### Overall Architecture
<p align="center">
  <img src="https://github.com/mishaalkandapath/neuralTuringMachine/blob/main/notes/transarch.png" alt="self_attention" width=50% />
</p><br>
The encoders all stacked together, finally producing the feature matrix which is fed into the decoder layers.

### Additional Information
1. Softmax is used on the decoder output to probabilities and a learned linear transformation.
2. The weights of this linear transformation is shared with the transformations used in the embedding layers to convert input and output tokens to vectors of dimension $d_{model}$. It is multiplied by the square of dimension in the embedding layers.
3. Since this architecture is not recurrent, some positional information is inserted. <b> Positional Encodings </b> are added to the inputs to the input and output embedders. The positional encodings are of the same dimensions, so they can be summed. <br>
<p align="center">
    $$ PE_{(pos, 2i)} = sin (pos/1000^{2i/d_{model}}) $$ 
</p>
for 2i+1, replace with cos. <br>
i represents the dimension in the vector. <br>
A learned positional encoding was not preferred due to similar performances, and also these sinusoidal waves can help generalize to lengtsh longer than encountered in training. 
