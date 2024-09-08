import jax
import jax.numpy as jnp

class RandomVariable():
  def __init__(self, name, coords, shape):
    self.name = name
    self.coords = coords
    self.shape = shape

class RV_w(RandomVariable):
  def __init__(self, alpha_w, beta_w, shape):
    coords = {'w_dim_0': [0]}
    super().__init__('w', coords, shape)
    self.alpha_w = alpha_w
    self.beta_w = beta_w

  def prior(self, key):
    return jax.random.beta(key, self.alpha_w, self.beta_w, shape=self.shape)

  def full_conditional_posterior(self, carry, key):
    conditional_alpha_w = jnp.sum(carry['T']) + carry['alpha_w']
    conditional_beta_w =  jnp.sum(1 -carry['T']) + carry['beta_w']
    return jax.random.beta(key, conditional_alpha_w, conditional_beta_w, shape=carry['w'].shape)

  def get_parameters(self):
    return {'alpha_w': self.alpha_w,
            'beta_w': self.beta_w}

class RV_p(RandomVariable):
  def __init__(self, alpha_p, beta_p, shape):
    num_experts = shape[0]
    coords = {'p_dim_0': range(num_experts)}
    super().__init__('p', coords, shape)
    self.alpha_p = jnp.broadcast_to(jnp.array(alpha_p),num_experts)
    self.beta_p = jnp.broadcast_to(jnp.array(beta_p),num_experts)

  def prior(self, key):
    return jax.random.uniform(key, minval=0.5, maxval=1, shape=self.shape)

  def full_conditional_posterior(self, carry, key):
    all_axis_except_last_one = tuple(range(carry["D"].ndim - 1))
    alphas_p = jnp.sum(carry["D"]*carry["T"], axis= all_axis_except_last_one) + carry["alpha_p"]
    betas_p = jnp.sum((1-carry["D"])*carry["T"], axis= all_axis_except_last_one) + carry['beta_p']
    return jax.random.beta(key, alphas_p, betas_p)

  def get_parameters(self):
    return {'alpha_p': self.alpha_p,
            'beta_p': self.beta_p}

class RV_q(RandomVariable):
  def __init__(self, alpha_q, beta_q, shape):
    num_experts = shape[0]
    coords = {'q_dim_0': range(num_experts)}
    super().__init__('q', coords, shape)
    self.alpha_q = jnp.broadcast_to(jnp.array(alpha_q),num_experts)
    self.beta_q = jnp.broadcast_to(jnp.array(beta_q),num_experts)

  def prior(self, key):
    return jax.random.uniform(key, minval=0.5, maxval=1, shape=self.shape)

  def full_conditional_posterior(self, carry, key):
    all_axis_except_last_one = tuple(range(carry["D"].ndim - 1))
    alphas_q = jnp.sum((1-carry["D"])*(1-carry["T"]), axis= all_axis_except_last_one) + carry['alpha_q']
    betas_q = jnp.sum(carry["D"]*(1-carry["T"]), axis= all_axis_except_last_one) + carry['beta_q']
    return  jax.random.beta(key, alphas_q, betas_q)
  
  def get_parameters(self):
    return {'alpha_q': self.alpha_q,
            'beta_q': self.beta_q}

class RV_T(RandomVariable):
  def __init__(self, w, shape):
    coords = {f'T_dim_{idx}':range(dim) for idx, dim in enumerate(shape)}
    super().__init__('T', coords, shape)
    self.w = w

  def prior(self, key):
    return jax.random.bernoulli(key, 0.5, shape=self.shape)

  def full_conditional_posterior(self, carry, key):
    numerator = jnp.prod(carry["D"]*carry["p"] + (1-carry["D"])*(1-carry["p"]), axis=(-2,-1), keepdims=True)*carry["w"]
    denominator = numerator + jnp.prod(carry["D"]*(1-carry["q"]) + (1-carry["D"])*carry["q"], axis=(-2,-1), keepdims=True)*(1-carry["w"])
    bernoulli_success_probability = numerator/denominator
    return jax.random.bernoulli(key, bernoulli_success_probability.reshape(carry["T"].shape), shape=carry["T"].shape)

  def get_parameters(self):
    return {'w': self.w}