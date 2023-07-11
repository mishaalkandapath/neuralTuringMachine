import jax
from jax import numpy as jnp
from jax import make_jaxpr
from jax import grad, jit, vmap, pmap
from functools import partial
from jax.nn import softmax
import numpy as np
import sys
import matplotlib.pyplot as plt
import progressbar
# jax numpy is a drop in replacement, meaning you can use almost everything the same way. for example:"
# jnp.arange(10), instead of np.arange(10); fairly intuitive. 

# the difference is in the datatypes you are actually dealing with: DeviceArrays (immutable unfort, unlike np.ndarray)
# to make differences, use var_.at[indx].set(new_val) (not a mutatiion, but a copy to replace operation)

#grad: gives a gradient function for a certain functioin
# jit: just in time compilation, for speedup
# vmap: vectorize a function, so that it can be applied to a batch of inputs without loops
# pmap: parallelize a function, so that it can be applied to a batch of inputs using many devices

# all of these are used for "functional transformations": akes a function and transforms it into another
# but only to be used for pure functions: 
#  - all inputs come in through the parameters only
#  - all outputs come out through the return value only
#  - same inputs = sam,e outsputs, always
#  for this, the objects you are passing in must be stateless
# in jax, for state updations, normally a new kobject is created in the state updation function, instead of updating it straight up

#  jaxpr: jax expression, a data structure that represents a function (an intermediate language really)
#  jax.jit:wrapping a function with jit  makes it run faster by optimiziung the jaxpr code. Run the fiunctiion once for compilation
# alternatively, by using the @jit decorator, we can pre-specify that the fucntion is to be jit compiled
# we cannot use jit wrapping for every function we have: 
# for example  a function with if and else's may have issues with tracing
# to get around this, make sure you are always having pure functions (eg somtimes if can be dewalth with jnp.where)

#jax vmap:
# takes a function, an in_axes parameter that specifies which input axes to map over, and an out_axes parameter that specifies where the mapped axis should appear in the output
# returns a new function
#some constants
char_to_idx = {}
idx_to_char = {}
key = jax.random.PRNGKey(0) #central randomness key

mode = "generation"
dmodel = 65#dimension of the embedding
dk, dq, dv = 4, 4, 4 # for now, test numbers. 
num_heads = 16
num_layers = 4
block_size = 32
batch_size = 16
dropout_prob = 0.2

jnp.set_printoptions(threshold=sys.maxsize)

@jit
def dropout(W):
    W *= np.random.binomial([np.ones(W.shape)],1-dropout_prob)[0] * (1.0/(1-dropout_prob))
    return W

#Remember to write the bit on :
#  - @partial
# - self attention masking and that our embeddin is of vectors of size 1
@jit
def scaled_dot_attention(queries, keys, values):
    # the scaling factor is to protect softmax from exploding, giving us trashy gradients. 
    # so we scale by dimension as magnitudes prop. dimension?
    #queries are the outputs from the decoder, keys and values are from the encoder. 
    
    vals = jnp.matmul(queries, keys.T)
    vals = jnp.where(jnp.tril(vals) == 0, -jnp.inf, vals) #mask out illegal positions
    dim_arr = jnp.asarray([queries.shape[-1]])
    vals = vals/(jnp.sqrt(dim_arr)[0])
    compatibilities = softmax(vals, axis=-1)
    compatibilities = dropout(compatibilities)
    return compatibilities @ values 

@jit
def multihead_attention(I, weights_q, weights_k, weights_v, weight_o):
    global dv, num_heads
    # the queries and keys are of dimension dmodel (the embedding dimension)
    # weights_q, k, and v are 3 dimensional matrices of dimension num_heads x dmodel x dk (dv)
    #weight_o is of size num_heads*dv x dmodel, and converts the concatenated outputs of the heads into the dmodel dimension - a linear transformation
    #not parallelizing, i dont got the hardware for that
    batched_scaled_dot_attention = vmap(scaled_dot_attention, in_axes=(0, 0, 0))
    batched_queries = I @ weights_q # vectorize over the heads dimensions
    batched_keys = I @ weights_k
    batched_values = I @ weights_v
    batched_attention = batched_scaled_dot_attention(batched_queries, batched_keys, batched_values) # doesnt have to be batched because of in_axes
    return dropout(batched_attention.reshape((block_size, num_heads*dv))) @ weight_o #dv, dk, dmodel/h = 64
                                     
@jit
def layer_norm(X, G, b, eps=1e-6):
    #UNDERSTAND WHY THIS WORKS - for now, something to do with computational time, exploding gradients or something
    mean_mat = jnp.mean(X, axis=-1, keepdims=True) #keepdims for proper broadcasting
    std_mat = jnp.var(X, axis=-1, keepdims=True)
    return G*(X-mean_mat)/jnp.sqrt(std_mat+eps) + b # G and B are learnable to tone down normalization when necessary

@jit
def ffn(X, W1, b1, W2, b2):
    return dropout(jnp.maximum(0, X @ W1 + b1) @ W2 + b2)

# important to have pure functions
@jit
def encoder(I, Wq, Wk, Wv, 
            persp_Wq, persp_Wk, persp_Wv, persp_Wo, 
            G1, b1, G2, b2,
            W_ff1, W_ff2, b_ff1, b_ff2):
    """
    An input is passed in which is a matrix, each row is a vector belonging to one token - its the token embedding or the output of the previous layer
    - each vector is of dimension dmodel=512 in paper. 
    
    @param I: the input matrix of tokens, each row is a token embedding
    @param Wq: the weight matrix for the queries, of dimension dmodel x dmodel
    @param Wk: the weight matrix for the keys, of dimension dmodel x dmodel
    @param Wv: the weight matrix for the values, of dimension dmodel x dmodel
    @param persp_Wq: the weight matrix for the queries, of dimension dmodel x dk
    @param persp_Wk: the weight matrix for the keys, of dimension dmodel x dk
    @param persp_Wv: the weight matrix for the values, of dimension dmodel x dv
    @param persp_Wo: the weight matrix for the output, of dimension hdv x dmodel
    @param G1: the gain matrix for the first layer norm, of dimension 512 x dmodel
    @param b1: the bias matrix for the first layer norm, of dimension 512 x 1
    @param G2: the gain matrix for the second layer norm, of dimension 512 x dmodel
    @param b2: the bias matrix for the second layer norm, of dimension 512 x 1
    @param W_ff1: the first weight matrix for the feed forward network, of dimension dmodel x dff
    @param W_ff2: the second weight matrix for the feed forward network, of dimension dff x dmodel
    @param b_ff1: the first bias matrix for the feed forward network, of dimension dff x 1
    @param b_ff2: the second bias matrix for the feed forward network, of dimension dmodel x 1
    """
    # first component the multi head self-attention 
    #it takes as input some queries, keys and values that you produce using linear transformations on the input I
    # the queries, keys and values are of dimension dmodel
    #layer norm 
    O = layer_norm(O, G1, b1)
    Queries, Keys, Values = I @ Wq, I @ Wk, I @ Wv #where I is the matrix of input tokens
    O = multihead_attention(Queries, Keys, Values, persp_Wq, persp_Wk, persp_Wv, persp_Wo) + I

    #feed forward network
    O = layer_norm(O, G2, b2)
    O = ffn(O, W_ff1, b_ff1, W_ff2, b_ff2) + O

    return O

@jit
def decoder_generate(I, 
            persp_Wq, persp_Wk, persp_Wv, persp_Wo,
            G1, b1, G2, b2, 
            W_ff1, W_ff2, b_ff1, b_ff2):
    """
    @param I: the input matrix of output tokens, each row is a token embedding
    @param enc_out: the output of the encoder, of dimension num_tokens x dmodel
    @param Wq: the weight matrix for the queries, of dimension dmodel x dmodel
    @param Wk: the weight matrix for the keys, of dimension dmodel x dmodel
    @param Wv: the weight matrix for the values, of dimension dmodel x dmodel
    
    the next three are used to pay attention to the encoder output using the decoder output as a query 
    @param W_dec_q: the weight matrix for the queries, of dimension dmodel x dmodel
    @param W_enc_k: the weight matrix for the keys, of dimension dmodel x dmodel
    @param W_enc_v: the weight matrix for the values, of dimension dmodel x dmodel

    @param persp_Wq: the weight matrix for the queries, of dimension dmodel x dk
    @param persp_Wk: the weight matrix for the keys, of dimension dmodel x dk
    @param persp_Wv: the weight matrix for the values, of dimension dmodel x dv

    perspective matrices for encoder attention using decoder queries
    @param persp_dec_Wq: the weight matrix for the queries, of dimension dmodel x dk
    @param persp_enc_Wk: the weight matrix for the keys, of dimension dmodel x dk
    @param persp_enc_Wv: the weight matrix for the values, of dimension dmodel x dv

    the rest are the same, refer encoder.
    """
    # lol i wonder what happens if i do attention over decoder outputs instead of on encoder ouputs
    # i know there is repeated code, I will rearrange in a while 
    O = multihead_attention(I, persp_Wq, persp_Wk, persp_Wv, persp_Wo) + I
    #layer norm 
    O = layer_norm(O, G1, b1)

    #feed forward network
    O = ffn(O, W_ff1, b_ff1, W_ff2, b_ff2) + O
    O = layer_norm(O, G2, b2)

    return O

def transformer_init(max_tokens=32, num_layers=4, num_heads=4, dmodel=32, dff=64, dk=4, dv=4):
    #returns all the matrices needed for the transformer with the mentioned number of layers and everything 
    #note: the positional encodings and the embedding is not included here
    #the encoder matrices first:
    global key
    rand_key, split1 = jax.random.split(key)
    rand_key, split2 = jax.random.split(rand_key)
    rand_key, split3 = jax.random.split(rand_key)
    enc_WQ, enc_WK, enc_WV = jax.random.normal(split1, (max_tokens, dmodel, num_layers)), jax.random.normal(split2, (max_tokens, dmodel, num_layers)), jax.random.normal(split3, (max_tokens, dmodel, num_layers))

    rand_key, split4 = jax.random.split(rand_key)
    rand_key, split5 = jax.random.split(rand_key)
    rand_key, split6 = jax.random.split(rand_key)
    #perspective matrices
    enc_persp_WQ, enc_persp_WK, enc_persp_WV = jax.random.normal(split4, (max_tokens, dk, num_heads, num_layers)), jax.random.normal(split5, (max_tokens, dk, num_heads, num_layers)), jax.random.normal(split6, (max_tokens, dv, num_heads, num_layers))

    rand_key, split7 = jax.random.split(rand_key)
    rand_key, split8 = jax.random.split(rand_key)
    rand_key, split9 = jax.random.split(rand_key)
    rand_key, split10 = jax.random.split(rand_key)
    # gains and biases for layer norm (encoder)
    enc_G1, enc_b1, enc_G2, enc_b2 = jax.random.normal(split7, (max_tokens, dmodel, num_layers)), jax.random.normal(split8, (dmodel, 1, num_layers)), jax.random.normal(split9, (max_tokens, dmodel, num_layers)), jax.random.normal(split10, (dmodel, 1, num_layers))

    rand_key, split11 = jax.random.split(rand_key)
    rand_key, split12 = jax.random.split(rand_key)
    rand_key, split13 = jax.random.split(rand_key)
    rand_key, split14 = jax.random.split(rand_key)
    # feed forward network matrices
    enc_W_ff1, enc_W_ff2, enc_b_ff1, enc_b_ff2 = jax.random.normal(split11, (max_tokens, dff, num_layers)), jax.random.normal(split12, (max_tokens, dmodel, num_layers)), jax.random.normal(split13, (dff, 1, num_layers)), jax.random.normal(split14, (dmodel, 1, num_layers))

    #decoder matrices
    rand_key, split15 = jax.random.split(rand_key)
    rand_key, split16 = jax.random.split(rand_key)
    rand_key, split17 = jax.random.split(rand_key)
    dec_WQ, dec_WK, dec_WV = jax.random.normal(split15, (max_tokens, dmodel, num_layers)), jax.random.normal(split16, (max_tokens, dmodel, num_layers)), jax.random.normal(split17, (max_tokens, dmodel, num_layers))

    rand_key, split18 = jax.random.split(rand_key)
    rand_key, split19 = jax.random.split(rand_key)
    rand_key, split20 = jax.random.split(rand_key)
    #perspective matrices
    dec_persp_WQ, dec_persp_WK, dec_persp_WV = jax.random.normal(split18, (max_tokens, dk, num_heads, num_layers)), jax.random.normal(split19, (max_tokens, dk, num_heads, num_layers)), jax.random.normal(split20, (max_tokens, dv, num_heads, num_layers))

    rand_key, split21 = jax.random.split(rand_key)
    rand_key, split22 = jax.random.split(rand_key)
    rand_key, split23 = jax.random.split(rand_key)
    rand_key, split24 = jax.random.split(rand_key)
    rand_key, split25 = jax.random.split(rand_key)
    rand_key, split26 = jax.random.split(rand_key)
    # gains and biases for layer norm (decoder)
    dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3 = jax.random.normal(split21, (max_tokens, dmodel, num_layers)), jax.random.normal(split22, (dmodel, 1, num_layers)), jax.random.normal(split23, (max_tokens, dmodel, num_layers)), jax.random.normal(split24, (dmodel, 1, num_layers)), jax.random.normal(split25, (max_tokens, dmodel, num_layers)), jax.random.normal(split26, (dmodel, 1, num_layers))

    rand_key, split27 = jax.random.split(rand_key)
    rand_key, split28 = jax.random.split(rand_key)
    rand_key, split29 = jax.random.split(rand_key)
    rand_key, split30 = jax.random.split(rand_key)
    # feed forward network matrices
    dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2 = jax.random.normal(split27, (max_tokens, dff, num_layers)), jax.random.normal(split28, (max_tokens, dmodel, num_layers)), jax.random.normal(split29, (dff, 1, num_layers)), jax.random.normal(split30, (dmodel, 1, num_layers))

    #decoder-encoder attention matrices
    rand_key, split31 = jax.random.split(rand_key)
    rand_key, split32 = jax.random.split(rand_key)
    rand_key, split33 = jax.random.split(rand_key)
    dec_enc_WQ, dec_enc_WK, dec_enc_WV = jax.random.normal(split31, (max_tokens, dmodel, num_layers)), jax.random.normal(split32, (max_tokens, dmodel, num_layers)), jax.random.normal(split33, (max_tokens, dmodel, num_layers))

    #decoder-enc persp matrices
    rand_key, split34 = jax.random.split(rand_key)
    rand_key, split35 = jax.random.split(rand_key)
    rand_key, split36 = jax.random.split(rand_key)
    dec_enc_persp_WQ, dec_enc_persp_WK, dec_enc_persp_WV = jax.random.normal(split34, (max_tokens, dk, num_heads, num_layers)), jax.random.normal(split35, (max_tokens, dk, num_heads, num_layers)), jax.random.normal(split36, (max_tokens, dv, num_heads, num_layers))

    #perspective output matrices for encoder and decoder
    rand_key, split37 = jax.random.split(rand_key)
    rand_key, split38 = jax.random.split(rand_key)
    rand_key, split39 = jax.random.split(rand_key)
    enc_persp_WO, dec_persp_WO, dec_enc_persp_WO = jax.random.normal(split37, (max_tokens, dmodel, num_heads, num_layers)), jax.random.normal(split38, (max_tokens, dmodel, num_heads, num_layers)), jax.random.normal(split39, (max_tokens, dmodel, num_heads, num_layers))

    #return all these matrices
    return enc_WQ, enc_WK, enc_WV, enc_persp_WQ, enc_persp_WK, enc_persp_WV, enc_G1, enc_b1, enc_G2, enc_b2, enc_W_ff1, enc_W_ff2, enc_b_ff1, enc_b_ff2, dec_WQ, dec_WK, dec_WV, dec_persp_WQ, dec_persp_WK, dec_persp_WV, dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3, dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, dec_enc_WQ, dec_enc_WK, dec_enc_WV, dec_enc_persp_WQ, dec_enc_persp_WK, dec_enc_persp_WV, enc_persp_WO, dec_persp_WO, dec_enc_persp_WO

def transformer_forward_encoder(X, enc_WQ, enc_WK, enc_WV, enc_persp_WQ, enc_persp_WK, enc_persp_WV, enc_G1, enc_b1, enc_G2, enc_b2, enc_W_ff1, enc_W_ff2, enc_b_ff1, enc_b_ff2, enc_persp_WO, num_layers=6):
    #unpack params
    # enc_WQ, enc_WK, enc_WV, enc_persp_WQ, enc_persp_WK, enc_persp_WV, enc_G1, enc_b1, enc_G2, enc_b2, enc_W_ff1, enc_W_ff2, enc_b_ff1, enc_b_ff2, enc_persp_WO = encoder_params

    #pass input through the encoders:
    prev_out = X # we start with the input as the one to feed into 
    assert type(num_layers) is int, "num_layers must be an integer"
    for i in range(num_layers):
        prev_out = encoder(prev_out, enc_WQ[:,:, i], enc_WK[:,:, i], enc_WV[:,:, i], 
                           enc_persp_WQ[:,:,:, i], enc_persp_WK[:,:,:, i], enc_persp_WV[:,:,:, i], enc_persp_WO[:,:, i], 
                           enc_G1[:,:, i], enc_b1[:,:, i], enc_G2[:,:, i], enc_b2[:,:, i],
                            enc_W_ff1[:,:, i], enc_W_ff2[:,:, i], enc_b_ff1[:,:, i], enc_b_ff2[:,:, i])
    #pass final output through the decoders:
    # oo think about a heirarchial transformer - where your predicting outputs by category level - what kind of sentence - what kind of word then - what is the word? etc

    return prev_out

@jit
def forward_pass_decoder_generate(prev_out, dec_persp_WQ, dec_persp_WK, dec_persp_WV,
                                   dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3,
                                     dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, dec_persp_WO, final_linear):
    # would it be possible to have the decoder be tuned per output, yea it doesnt make a difference hmm
    #unpack params
    global num_layers
    for i in range(num_layers):
        prev_out = decoder_generate(prev_out, 
                        dec_persp_WQ[:,:, :,i], dec_persp_WK[:,:, :, i], dec_persp_WV[:,:, :, i], dec_persp_WO[:,:, i],
                        dec_G1[:,:, i], dec_b1[:,:, i], dec_G2[:,:, i], dec_b2[:,:, i],
                        dec_W_ff1[:,:, i], dec_W_ff2[:,:, i], dec_b_ff1[:,:, i], dec_b_ff2[:,:, i])
    #add linear and softmax
    prev_out = layer_norm(prev_out, dec_G3, dec_b3)
    prev_out = softmax(prev_out @ final_linear)
    return prev_out

@partial(jit, static_argnums=(2,))
def adam(grad, weight, iters=0, beta1 = 0.9, beta2 = 0.999, m=0,v=0, lr=0.001):
    #clip the norms of the gradients
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    mhat = m / (1 - beta1 ** (iters + 1))
    vhat = v / (1 - beta2 ** (iters + 1))
    # lr = (dmodel ** (-0.5)) * min(iters**(-0.5), iters * (2000 ** (-1.5)))
    weight = weight - lr * mhat / (jnp.sqrt(vhat) + 1e-9)
    return weight, m, v

@jit
def forward_loss_generate(X, Y,
                           dec_persp_WQ, dec_persp_WK, dec_persp_WV,
                             dec_G1, dec_b1, dec_G2, dec_b2,dec_G3, dec_b3, 
                               dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, 
                          dec_persp_WO, final_linear):
    
    def non_vectorized(x, y):
        probs = forward_pass_decoder_generate(x, dec_persp_WQ, dec_persp_WK, dec_persp_WV,
                             dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3,
                               dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, 
                          dec_persp_WO, final_linear)
        loss = jnp.sum(-jnp.log(probs) * y)
        return loss, probs

    batched_losses, batched_probs = vmap(non_vectorized)(X, Y)
    loss = jnp.sum(batched_losses)
        
    return loss, batched_probs

#this is largely wrong, will be corrected later.
def transformer_train_translation(X, Y, encoder_params, decoder_params, final_linear, iters):
    # global num_layers
    # #here am resolving to compute loss at the end of sequence generation for now.
    # assert type(num_layers) is int, "num_layers must be an integer"
    # argnums = [3 + i for i in range(len(encoder_params))] + [3 + len(encoder_params) + i for i in range(len(decoder_params))] + [3 + len(encoder_params) + len(decoder_params)]

    # for _ in range(iters):
    #     #pass a forward pass, and ADAM it
    #     enc_WQ, enc_WK, enc_WV, enc_persp_WQ, enc_persp_WK, enc_persp_WV, enc_G1, enc_b1, enc_G2, enc_b2, enc_W_ff1, enc_W_ff2, enc_b_ff1, enc_b_ff2, enc_persp_WO = encoder_params
    #     dec_WQ, dec_WK, dec_WV, dec_persp_WQ, dec_persp_WK, dec_persp_WV, dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3, dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, dec_enc_WQ, dec_enc_WK, dec_enc_WV, dec_enc_persp_WQ, dec_enc_persp_WK, dec_enc_persp_WV, dec_persp_WO, dec_enc_persp_WO = decoder_params
    #     #pass input through the encoders:
    #     prev_out = [-jnp.inf] * (max_tokens) # replace None with the start token 
    #     prev_out = jnp.asarray(prev_out)
    #     loss, prev_out, grads = jax.value_and_grad(forward_loss, argnums=argnums)(X, Y, prev_out,
    #                                                 enc_WQ, enc_WK, enc_WV, enc_persp_WQ, enc_persp_WK, enc_persp_WV, enc_G1, enc_b1, enc_G2, enc_b2, enc_W_ff1, enc_W_ff2, enc_b_ff1, enc_b_ff2, enc_persp_WO, 
    #                                                 dec_WQ, dec_WK, dec_WV, dec_persp_WQ, dec_persp_WK, dec_persp_WV, dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3, dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, 
    #                                                 dec_enc_WQ, dec_enc_WK, dec_enc_WV, dec_enc_persp_WQ, dec_enc_persp_WK, dec_enc_persp_WV, dec_persp_WO, dec_enc_persp_WO, 
    #                                                 final_linear, num_layers, max_tokens)
        
    #     #update the params using ADAM
    #     encoder_params = [adam(grads[i], encoder_params[i]) for i in range(len(encoder_params))]
    #     decoder_params = [adam(grads[i + len(encoder_params)], decoder_params[i]) for i in range(len(decoder_params))]
    #     final_linear = adam(grads[-1], final_linear)        
            
    #     if _ % 100 == 0:
    #         print(decode(prev_out))
    #         print(loss)
    pass # for now


def transformer_train_generation(X, Y, decoder_params, final_linear, iters):
    global num_layers, char_to_idx
    dec_persp_WQ, dec_persp_WK, dec_persp_WV, dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3, dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, dec_persp_WO = decoder_params
    #prepare the X array, where X here is a matrix of examples only , make them to be one at a time
    #make a new dimension, the dimension of training examples:
    if len(X.shape) == 2:
        # #extract each of the second dimension into the first dimension, and make the second dimension a single element 
        #insert a new column of zeros at the start of columns
        # X = np.insert(X, 0, 0, axis=1)
        X_new = jnp.eye(dmodel)[X]
        X=X_new
        Y_new = jnp.eye(dmodel)[Y] 
        Y = Y_new
        #add positional encodings
        X = vmap(encode_with_positional, in_axes=0)(X)
    #otherwise can assume it came in the right shape, coz we need to parallelize over the training examples
    #change this, assuming that there is only one example that is being passed through:
    argnums = [2+i for i in range(len(decoder_params)+1)]
    prev_grads = [[]] * 15
    prev_params = [[]] * 15
    m=[0]*15
    v=[0]*15
    losses=[]
    count=1
    for _ in progressbar.progressbar(range(iters*100), redirect_stdout=True):
        # for j in range(X.shape[0]):
        # x, y = X[j], Y[j]
        #here the thing is, given our encoding dmodel = 1, thats kinda shet lol
        (loss, out), grads = jax.value_and_grad(forward_loss_generate, argnums=argnums, has_aux=True)(X, Y, dec_persp_WQ, dec_persp_WK, dec_persp_WV, dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3, dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, dec_persp_WO, final_linear)
        assert not jnp.isnan(loss) or prev_grads is not None, "loss is nan in the very beginning, reinit"
        losses+=[loss]

        #write all the parameters to their respective files:
        
        # param_names = ['persp_WQs', 'persp_WKs', 'persp_WVs',  'G1s', 'b1s', 'G2s', 'b2s', 'Wff1s', 'Wff2s',  'bff1s', 'bff2s', 'persp_WOs', 'final_lins']
        # for i in range(len(decoder_params)):
        #     jnp.save("weights/"+param_names[i]+"/"+str((_))+"_"+str(j), decoder_params[i])
        # jnp.save("weights/"+param_names[-1]+"/"+str((_))+"_"+str(j), final_linear)
        decoder_params = [adam(grads[k], decoder_params[k], count, m=m[k], v=v[k]) for k in range(len(decoder_params))]
        for k in range(len(decoder_params)):
            m[k] = decoder_params[k][1]
            v[k] = decoder_params[k][2]
            decoder_params[k] = decoder_params[k][0]
        final_linear, m[-1], v[-1] = adam(grads[-1], final_linear, count, m=m[-1], v=v[-1])
        count+=1
        if _ % 500 == 0 and _ != 0:
            print(decode(out[-1]))
            print(decode(Y[-1]))
            # print(len(losses))
            plt.plot(losses)
            plt.savefig("losses.png")
            # assert not jnp.isnan(loss), "loss is nann, stopping"
            param_names = ['persp_WQs', 'persp_WKs', 'persp_WVs',  'G1s', 'b1s', 'G2s', 'b2s', 'Wff1s', 'Wff2s',  'bff1s', 'bff2s', 'persp_WOs', 'final_lins']
            for idx,param in enumerate(param_names[:-1]):
                if jnp.isnan(decoder_params[idx]).any():
                    print(param + " is empty")#, [decoder_params + [final_linear]][idx])
            assert not any(jnp.isnan(param).any() for param in decoder_params), "wooops"

def routine_start(mode):
    global char_to_idx, idx_to_char, batch_size, key
    assert mode in ["translation", "generation"], "invalid option, choose among 'translation' and 'generation'"
    #first take in the data
    f = open("input.txt")
    chars = sorted(list(set(f.read())))
    f.seek(0)
    # chars = ['<PAD>'] + chars
    char_to_idx = {ch:i for i, ch in enumerate(chars)}
    idx_to_char = {i:ch for i, ch in enumerate(chars)} # encoder and decoder mappings,

    #lets get the data:
    data = f.read()
    data = jnp.asarray(encode(data))

    #spl9it data into train and test
    train_data, test_data = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]

    train_x, train_y = data_loader(train_data)
    test_x, test_y = data_loader(test_data)

    #time for training. 
    #init params
    if mode == "translation":
        encoder_params = init_encoder_params()
    decoder_params = init_decoder_params()

    # key, final_split = jax.random.split(key)
    # final_linear_trans = jax.random.normal(final_split, (dmodel, len(char_to_idx.keys())))
    final_linear_trans = np.random.normal(size=(dmodel, len(char_to_idx.keys())))

    if mode == "translation":
        transformer_train_translation(train_x, train_y, encoder_params, decoder_params, final_linear_trans, 1000)
    else:
        transformer_train_generation(train_x, train_y, decoder_params, final_linear_trans, 1000)

def encode(text):
    global char_to_idx
    return [char_to_idx[ch] for ch in text]

def encode_with_positional(sequence_matrix):
    global dmodel
    denoms = jnp.asarray([10000 ** ((i//2)/dmodel) for i in range(dmodel)])
    #repeat denoms for each row, for block_size nmber of times
    denoms = jnp.expand_dims(denoms, axis=0)
    denoms = jnp.repeat(denoms, block_size, axis=0)
    # print(denoms)
    positions = jnp.arange(block_size)
    positions = jnp.expand_dims(positions, axis=0)
    positions = jnp.repeat(positions.T, dmodel, axis=1)
    positions = positions.astype(jnp.float32)
    
    positions = positions.at[:,::2].set(jnp.sin(positions / denoms)[:, ::2])
    positions = positions.at[:,1::2].set(jnp.cos(positions / denoms)[:, 1::2])
    sequence_matrix = jnp.asarray(sequence_matrix, dtype=jnp.float32)
    sequence_matrix += positions
    
    return sequence_matrix

def decode(arr):
    global idx_to_char
    #get indices of 
    # arr = jnp.squeeze(arr)
    col_dices = jnp.argmax(arr, axis=1)
    indices = jnp.arange(len(col_dices))
    arr = arr[indices, col_dices]
    arr = col_dices
    arr = arr.tolist()
    
    return "".join([idx_to_char[i] for i in arr])

def init_encoder_params():
    global key, block_size, num_layers, num_heads, dk, dv
    splits = jax.random.split(key, 15)
    key = splits[0]
    splits = splits[1:]
    #note, the output persp shape is because of the choice of dv = dmodel/h, due to which h * dv = dmodel 
    #refer below for matrix descriptions
    weights = [(num_heads, dmodel, dk, num_layers), (num_heads, dmodel, dk, num_layers), (num_heads, dmodel, dk, num_layers),
                  (block_size, dmodel, num_layers), (1, dmodel, num_layers), (block_size, dmodel, num_layers), (1, dmodel, num_layers),
                    (dmodel, dmodel * 4, num_layers), (dmodel * 4, dmodel, num_layers), (1, dmodel * 4, num_layers), (1, dmodel, num_layers),
                         (num_heads * dv, block_size, num_layers)]
    for idx in range(len(weights)):
        weights[idx] = jax.random.normal(splits[idx], weights[idx])
    
    return tuple(weights)

def init_decoder_params():
    global key, block_size, num_layers, num_heads, dk, dv, dq, dmodel
    splits = jax.random.split(key, 24)
    key = splits[0]
    splits = splits[1:]
    # order: Wq, Wk, Wv (removed these three),
    # persp_Wq, persp_Wk, persp_Wv,  
    # G1, b1, G2, b2, G3, b3,
    # W_ff1, W_ff2, b_ff1, b_ff2, 
    # dec_Wq, enc_Wk, enc_Wv, dec_persp_Wq, enc_persp_Wk (not there for generation), 
    # enc_persp_Wv, persp_WO, dec_persp_WO (not there for generation)
    #the first query etc matrices simply transform the embeddings within the same space - not needed, can be learnt directly by the multihead mechanism, so removing (coz its a linearity).
    #the perspectives put them onto another space
    weights = [(num_heads, dmodel, dk, num_layers), (num_heads, dmodel, dk, num_layers), (num_heads, dmodel, dk, num_layers),
                  (block_size, dmodel, num_layers), (block_size, dmodel, num_layers), (block_size, dmodel, num_layers), (block_size, dmodel, num_layers), (block_size, dmodel), (block_size, dmodel),
                    (dmodel, dmodel * 4, num_layers), (dmodel * 4, dmodel, num_layers), (1, dmodel * 4, num_layers), (1, dmodel, num_layers),
                         (num_heads * dv, dmodel, num_layers)]
    
    if mode == "translation":
        #make sure to add these at the right position
        pass #here you can add the other matrices for the encoder decoder attention block and layer norm
    limit = np.sqrt(3 / float(65*2))
    for idx in range(len(weights)):
        #non jaxifyhing randomization for now
        # weights[idx] = np.random.normal(size=weights[idx])
        if idx in range(3, 8) and idx % 2 == 1:
            weights[idx] = jnp.ones(weights[idx])
            continue
        elif idx in range(3,8) and idx % 2 == 0:
            weights[idx] = jnp.zeros(weights[idx])
            continue
        elif idx in range(2): #coz they are 4D 
            limit = np.sqrt(2 / float(weights[idx][1] + weights[idx][2]))
            # limit = np.sqrt(weights[idx][1])
        else:
            limit = np.sqrt(2/ float(weights[idx][0] + weights[idx][1]))
            # limit = limit = np.sqrt(weights[idx][0])
        weights[idx] = np.random.uniform(-limit, limit, size=weights[idx]) ##np.random.normal(0, limit, size=weights[idx])
        # weights[idx] = jax.random.normal(splits[idx], weights[idx])
    
    return tuple(weights)


def data_loader(data):
    global batch_size, block_size
    #chunk the data into batches
    #batchsize= 4
    #there block_size = 8, context vector, the lnegth of rthe sequence to predict
    global key
    indices = jax.random.randint(key, (batch_size, ), 0, len(data) - block_size)
    x = jnp.stack([data[idx:idx + block_size] for idx in indices])
    y = jnp.stack([data[idx+1:idx + block_size+1] for idx in indices])
    return x, y

if __name__ == "__main__":
    routine_start("generation")

    # final_lins = jnp.load("weights/persp_WQs/130_0.npy")
    # print(final_lins)

    
"""
At a certain point in time, you're computing one sequence type at a time. 
in that time, you are going through a matrix of all the parts of that sequence. 
so the main matrix is of size block_size * dmodel

Some problems currently occuring
 - Exploding Gradients I think ?

Talk about jax jit compilations and tracing and printing
"""