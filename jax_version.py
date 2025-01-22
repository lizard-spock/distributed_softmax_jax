import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

#create a single tensor
arr = jnp.arange(32.0).reshape(4, 8)
print(arr.devices())
print(jax.devices())

#shard it in 8 ways 
devices = np.array(jax.devices()).reshape(2, 4)
mesh = jax.sharding.Mesh(devices, ('x','y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
print(sharding)

arr_sharded = jax.device_put(arr, sharding)

print(arr_sharded)
# jax.debug.visualize_array_sharding(arr_sharded)

#1.Automated parallelism via jit
@jax.jit
def f_elementwise(x):
  return 2 * jnp.sin(x) + 1

result = f_elementwise(arr_sharded)

print("shardings match:", result.sharding == arr_sharded.sharding)
print("result device: ", result.devices())

@jax.jit
def f_contract(x):
  return x.sum(axis=0)

result = f_contract(arr_sharded)
print(result)
print("result device: ", result.devices())
jax.debug.visualize_array_sharding(result)