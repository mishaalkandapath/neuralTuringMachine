import numpy as np
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
