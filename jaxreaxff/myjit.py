from functools import partial

from jax import jit
import jax.numpy as jnp

def some_hash_function(x):
  return int(jnp.sum(x))

class HashableArrayWrapper:
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return some_hash_function(self.val)
  def __eq__(self, other):
    return (isinstance(other, HashableArrayWrapper) and
            jnp.all(jnp.equal(self.val, other.val)))

class HashableListArrayWrapper:
  MOD = 10**9 + 7
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    res = 0
    for i,sub_arr in enumerate(self.val):
      res += (some_hash_function(sub_arr)) % HashableListArrayWrapper.MOD
    return res
  def __eq__(self, other):
    if (isinstance(other, HashableListArrayWrapper) == False or 
      len(self.val) != len(other.val)):
      return False
    else:
      for i in range(len(self.val)):
        if jnp.all(jnp.equal(self.val[i], other.val[i])) == False:
          return False
    return True

def my_jit(fun, static_argnums=(), 
                static_array_argnums=(), 
                static_list_of_array_argnums=(),
                backend=None):

  def callee(*args):
    args = list(args)
    for i in (static_array_argnums + static_list_of_array_argnums):
      args[i] = args[i].val

    return fun(*args)

  def caller(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = HashableArrayWrapper(args[i])
    for i in static_list_of_array_argnums:
      args[i] = HashableListArrayWrapper(args[i])
    return jit(callee, backend=backend,
                   static_argnums=static_argnums + \
                   static_array_argnums + \
                   static_list_of_array_argnums)(*args)

  return caller
