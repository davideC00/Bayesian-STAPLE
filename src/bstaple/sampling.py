import xarray as xr
from jax_tqdm import scan_tqdm
import jax.numpy as jnp
import jax


class GibbsSampler():
  def __init__(self, vars, data, seed):
    self.vars = vars
    self.data = data
    self.seed = seed
    self.key = jax.random.key(self.seed)

  def initialize_carry(self, key):
    carry = self.data
    for var in self.vars:
      carry = {**carry, **var.get_parameters()}
    for var in self.vars:
      carry[var.name] = var.prior(key)
    
    return carry

  # Create the sampling function used by the jax.lax.scan
  def sampling_fun(self):
    def wrapper_sampling_fun(carry, tupl):
      _, key = tupl
      draw = {}
      for var in self.vars:
        value = var.full_conditional_posterior(carry, key)
        carry[var.name] = value
        draw[var.name] = value
      return (carry), (draw)
    return wrapper_sampling_fun

  def sample(self, iterations,  burn_in=0, chains=1):

    sampling_fun = self.sampling_fun()
    sampling_fun_tqdm = (scan_tqdm(iterations)(sampling_fun)) # progress bar
    sampling_fun_tqdm_jit = jax.jit(sampling_fun_tqdm)

    key = jax.random.key(self.seed)
    chains_keys = jax.random.split(key, chains)

    traces = []
    for i in range(chains):

      carry = self.initialize_carry(chains_keys[i])

      keys = jax.random.split(chains_keys[i], iterations)
      keys = (jnp.arange(iterations), keys)
      _, sample = jax.lax.scan(sampling_fun_tqdm_jit, carry, keys)

      chain_trace = {}
      for var in self.vars:
        coords =  {"draw": range(iterations), **var.coords}
        chain_trace[var.name] = xr.DataArray(sample[var.name], coords=coords).squeeze(drop=True)
      chain_trace = xr.Dataset(chain_trace)
      chain_trace = chain_trace.isel(draw=range(burn_in,iterations))
      chain_trace = chain_trace.expand_dims(dim={"chain": [i]}, axis=0)
      traces.append(chain_trace)

      # Release RAM between iterations
      # https://forum.pyro.ai/t/gpu-memory-preallocated-and-not-released-between-batches/3774
      # https://github.com/google/jax/issues/2072
      jax.block_until_ready(vars)
      jax.block_until_ready(carry)
      jax.block_until_ready(self.data)
      jax.block_until_ready(sample)
      jax.block_until_ready(keys)
      jax.block_until_ready(chain_trace)
      sampling_fun_tqdm_jit._clear_cache()
    data = xr.concat(traces, 'chain')
    return data


