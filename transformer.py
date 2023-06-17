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
def layer_norm(X, G, B, eps=1e-6):
    #UNDERSTAND WHY THIS WORKS - for now, something to do with computational time, exploding gradients or something
    mean_mat = jnp.mean(X, axis=-1, keepdims=True) #keepdims for proper broadcasting
    std_mat = jnp.std(X, axis=-1, keepdims=True)
    return G*(X-mean_mat)/(std_mat+eps) + B # G and B are learnable to tone down normalization when necessary

# important to have pure functions
@jit
def encoder(I, Wq, Wk, Wv, persp_Wq, persp_Wk, persp_Wv, persp_Wo, G1, B1, G2, B2):
    """
    An input is passed in which is a matrix, each row is a vector belonging to one token - its the token embedding or the output of the previous layer
    - each vector is of dimension dmodel=512 in paper. 
    """
    # first component the multi head self-attention 
    #it takes as input some queries, keys and values that you produce using linear transformations on the input I
    # the queries, keys and values are of dimension dmodel
    Queries, Keys, Values = I @ Wq, I @ Wk, I @ Wv #where I is the matrix of input tokens
    O = multihead_attention(Queries, Keys, Values, persp_Wq, persp_Wk, persp_Wv, persp_Wo) + I
    #layer norm 
    O = layer_norm(O, G1, B1)




         
        


    def decoder():
        pass