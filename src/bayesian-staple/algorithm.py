import jax.numpy as jnp
from jax import random
import xarray as xr
from jax_tqdm import scan_tqdm
from jax.lax import scan

class BayesianSTAPLE():

  def __init__(self, D, w=None,  alpha_p = 1, beta_p = 1, alpha_q=1, beta_q=1,
              alpha_w=1, beta_w=1,
              repeated_labeling = False, seed= 1701):
    self.seed = seed
    self.steps = []
    self.initialization_steps = []
    self.D = jnp.array(D, dtype='byte')
    if not (repeated_labeling): self.D = jnp.expand_dims(D, axis=-2)
    T_shape = list(self.D.shape)
    T_shape[-1] = 1
    T_shape[-2] = 1
    self.T_shape = tuple(T_shape)
    self.num_experts = D.shape[-1]
    self.coords = {}
    self.key = random.key(self.seed)

    self.vars = {}
    if w == None:
      # old behaviour: self.vars['w'] = D.sum()/D.size
      self.vars['alpha_w'] = jnp.array(alpha_w)
      self.vars['beta_w'] = jnp.array(beta_w)
      self.add_step_w()
    else:
      self.vars['w'] = jnp.array(w)

    self.vars['alpha_p'] = jnp.broadcast_to(jnp.array(alpha_p),self.num_experts)
    self.vars['beta_p'] = jnp.broadcast_to(jnp.array(beta_p),self.num_experts)
    self.vars['alpha_q'] = jnp.broadcast_to(jnp.array(alpha_q),self.num_experts)
    self.vars['beta_q'] = jnp.broadcast_to(jnp.array(beta_q),self.num_experts)

    self.add_step_T()
    self.add_step_p()
    self.add_step_q()

  def add_step_w(self):
    self.coords['w'] = {'w_dim_0': [0]}

    def initiallization_w(vars, key):
      vars['w'] = random.beta(key, vars['alpha_w'], vars['beta_w'], shape=(1,))

    self.initialization_steps.append(initiallization_w)

    def step_w( vars, key):
      conditional_alpha_w = jnp.sum(vars['T']) + vars['alpha_w']
      conditional_beta_w =  jnp.sum(1 -vars['T']) + vars['beta_w']
      vars['w']  = random.beta(key, conditional_alpha_w, conditional_beta_w, shape=(1,))

    self.steps.append(step_w)
    return self

  def add_step_p(self):
    self.coords['p'] = {'p_dim_0': range(self.num_experts)}

    def initiallization_p(vars, key):
      vars['p'] = random.uniform(key, minval=0.5, maxval=1, shape=(self.num_experts,))

    self.initialization_steps.append(initiallization_p)

    def step_p( vars, key):
      all_axis_except_last_one = tuple(range(self.D.ndim - 1))
      alphas_p = jnp.sum(self.D*vars["T"], axis= all_axis_except_last_one) + vars["alpha_p"]
      betas_p = jnp.sum((1-self.D)*vars["T"], axis= all_axis_except_last_one) + vars['beta_p']
      vars["p"] = random.beta(key, alphas_p, betas_p)

    self.steps.append(step_p)
    return self

  def add_step_q(self):
    self.coords['q'] = {'q_dim_0': range(self.num_experts)}

    def initiallization_q(vars, key):
      vars['q'] = random.uniform(key, minval=0.5, maxval=1, shape=(self.num_experts,))

    self.initialization_steps.append(initiallization_q)

    def step_q( vars, key):
      all_axis_except_last_one = tuple(range(self.D.ndim - 1))
      alphas_q = jnp.sum((1-self.D)*(1-vars["T"]), axis= all_axis_except_last_one) + vars['alpha_q']
      betas_q = jnp.sum(self.D*(1-vars["T"]), axis= all_axis_except_last_one) + vars['beta_q']
      vars["q"]  = random.beta(key, alphas_q, betas_q)

    self.steps.append(step_q)
    return self

  def add_step_T(self):
    self.coords["T"] = {f'T_dim_{idx}':range(dim) for idx, dim in enumerate(self.T_shape)}

    def initiallization_T(vars, key):
      vars['T'] = random.bernoulli(key, 0.5, shape=self.T_shape)

    self.initialization_steps.append(initiallization_T)

    def step_T(vars, key):
      numerator = jnp.prod(self.D*vars["p"] + (1-self.D)*(1-vars["p"]), axis=(-2,-1), keepdims=True)*vars["w"]
      denominator = numerator + jnp.prod(self.D*(1-vars["q"]) + (1-self.D)*vars["q"], axis=(-2,-1), keepdims=True)*(1-vars["w"])
      bernoulli_success_probability = numerator/denominator
      vars["T"]  = random.bernoulli(key, bernoulli_success_probability.reshape(self.T_shape), shape=self.T_shape)

    self.steps.append(step_T)
    return self

  # Create the sampling function used by the jax.lax.scan
  def sampling_fun(self):
    def wrapper_sampling_fun(vars, tupl):
      _, key = tupl
      for step in self.steps:
        step( vars, key)
      return (vars), (vars)

    return wrapper_sampling_fun

  def sample(self, draws,  burn_in=0, chains=1):

    sampling_fun = self.sampling_fun()
    sampling_fun_jit_tqdm = (scan_tqdm(draws)(sampling_fun)) # progress bar

    key = random.key(self.seed)
    chains_keys = random.split(key, chains)

    traces = []
    for i in range(chains):

      for step in self.initialization_steps:
        step(self.vars, chains_keys[i])

      keys = random.split(chains_keys[i], draws)
      keys = (jnp.arange(draws), keys)
      _, samples = jax.lax.scan(sampling_fun_jit_tqdm, self.vars, keys)

      # Release RAM between iterations
      # https://forum.pyro.ai/t/gpu-memory-preallocated-and-not-released-between-batches/3774
      for var in self.vars:
        jax.block_until_ready(var)
      jax.block_until_ready(self.D)

      chain_trace = {}
      for var in self.coords:
      #  if var in :  # vars not in coords are not saved in the dataset
        self.coords[var] =  {"draw": range(draws), **self.coords[var]}
        chain_trace[var] = xr.DataArray(samples[var], coords=self.coords[var]).squeeze(drop=True)
      chain_trace = xr.Dataset(chain_trace)
      chain_trace = chain_trace.isel(draw=range(burn_in, draws))
      chain_trace = chain_trace.expand_dims(dim={"chain": [i]}, axis=0)
      traces.append(chain_trace)
    data = xr.concat(traces, 'chain')
    return data

  def get_ground_truth(self, sample):
    return sample.T.mean(axis=self.T_shape[0:-2])


