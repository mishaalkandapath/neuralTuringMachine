import jax
from jax import numpy as jnp
from jax import make_jaxpr
from jax import grad, jit, vmap, pmap

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

@jit
def scaled_dot_attention(queries, keys, values):
    # the scaling factor is to protect softmax from exploding, giving us trashy gradients. 
    # so we scale by dimension as magnitudes prop. dimension?
    #queries are the outputs from the decoder, keys and values are from the encoder. 
    
    compatibilities = jnp.softmax(jnp.matmul(queries, keys.T)/jnp.sqrt(queries.shape[-1]))
    return compatibilities @ values 
@jit 
def multihead_attention(queries, keys, values, weights_q, weights_k, weights_v, weight_o, num_heads=8):
    # the queries and keys are of dimension dmodel (the embedding dimension)
    # weights_q, k, and v are 3 dimensional matrices of dimension num_heads x dmodel x dk (dv)
    #weight_o is of size num_heads*dv x dmodel, and converts the concatenated outputs of the heads into the dmodel dimension - a linear transformation
    #not parallelizing, i dont got the hardware for that
    batched_scaled_dot_attention = vmap(scaled_dot_attention)
    batched_queries = queries @ weights_q # vectorize over the heads dimensions
    batched_keys = keys @ weights_k
    batched_values = values @ weights_v
    batched_attention = batched_scaled_dot_attention(batched_queries, batched_keys, batched_values)
    return batched_attention.reshape((num_heads*64, 512)) @ weight_o #dv, dk, dmodel/h = 64
                                     
@jit
def layer_norm(X, G, b, eps=1e-6):
    #UNDERSTAND WHY THIS WORKS - for now, something to do with computational time, exploding gradients or something
    mean_mat = jnp.mean(X, axis=-1, keepdims=True) #keepdims for proper broadcasting
    std_mat = jnp.std(X, axis=-1, keepdims=True)
    return G*(X-mean_mat)/(std_mat+eps) + b # G and B are learnable to tone down normalization when necessary

@jit
def ffn(X, W1, b1, W2, b2):
    return jnp.maximum(0, X @ W1 + b1) @ W2 + b2

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
    Queries, Keys, Values = I @ Wq, I @ Wk, I @ Wv #where I is the matrix of input tokens
    O = multihead_attention(Queries, Keys, Values, persp_Wq, persp_Wk, persp_Wv, persp_Wo) + I
    #layer norm 
    O = layer_norm(O, G1, b1)

    #feed forward network
    O = ffn(O, W_ff1, b_ff1, W_ff2, b_ff2) + O
    O = layer_norm(O, G2, b2)

    return O

def decoder(I, enc_out, Wq, Wk, Wv, W_dec_q, W_enc_k, W_enc_v, 
            persp_Wq, persp_Wk, persp_Wv, persp_Wo, persp_dec_Wq, persp_enc_Wk, persp_enc_Wv, persp_dec_Wo, 
            G1, b1, G2, b2, G3, b3, 
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
    Queries, Keys, Values = I @ Wq, I @ Wk, I @ Wv #where I is the matrix of input tokens
    O = multihead_attention(Queries, Keys, Values, persp_Wq, persp_Wk, persp_Wv, persp_Wo) + I
    #layer norm 
    O = layer_norm(O, G1, b1)

    #attention using the encoder output using decoder values as key
    dec_Queries, dec_Keys, dec_Values = O @ W_dec_q, enc_out @ W_enc_k, enc_out @ W_enc_v
    O = multihead_attention(dec_Queries, dec_Keys, dec_Values, persp_dec_Wq, persp_enc_Wk, persp_enc_Wv, persp_dec_Wo) + O
    #layer norm
    O = layer_norm(O, G2, b2)

    #feed forward network
    O = ffn(O, W_ff1, b_ff1, W_ff2, b_ff2) + O
    O = layer_norm(O, G3, b3)

    return O

def transformer_init(max_tokens=512, num_layers=6, num_heads=8, dmodel=512, dff=2048, dk=64, dv=64):
    #returns all the matrices needed for the transformer with the mentioned number of layers and everything 
    #note: the positional encodings and the embedding is not included here
    #the encoder matrices first:
    rand_key = jax.random.PRNGKey(0)
    rand_key, split1 = jax.random.split(rand_key)
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
    return enc_WQ, enc_WK, enc_WV, enc_persp_WQ, enc_persp_WK, enc_persp_WV, enc_G1, enc_b1, enc_G2, enc_b2, enc_G3, enc_b3, enc_W_ff1, enc_W_ff2, enc_b_ff1, enc_b_ff2, dec_WQ, dec_WK, dec_WV, dec_persp_WQ, dec_persp_WK, dec_persp_WV, dec_G1, dec_b1, dec_G2, dec_b2, dec_G3, dec_b3, dec_W_ff1, dec_W_ff2, dec_b_ff1, dec_b_ff2, dec_enc_WQ, dec_enc_WK, dec_enc_WV, dec_enc_persp_WQ, dec_enc_persp_WK, dec_enc_persp_WV, enc_persp_WO, dec_persp_WO, dec_enc_persp_WO
    

