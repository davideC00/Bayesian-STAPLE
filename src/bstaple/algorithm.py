import jax.numpy as jnp
from .random_vars import RV_w, RV_T, RV_p, RV_q
from .sampling import GibbsSampler

class BayesianSTAPLE():

  def __init__(self, D, w=None,  alpha_p = 1, beta_p = 1, alpha_q=1, beta_q=1,
              alpha_w=1, beta_w=1,
              repeated_labeling = False, seed= 1701):
    D = jnp.array(D, dtype='byte')
    if not (repeated_labeling): D = jnp.expand_dims(D, axis=-2) # add void dimension to D

    random_vars = []

    T_shape = list(D.shape)
    T_shape[-1] = 1
    T_shape[-2] = 1
    T_shape = tuple(T_shape)
    if w == None:
      # hierarchical model
      w = RV_w(alpha_w, beta_w, shape=(1,)) 
      random_vars.append(w) 
      random_vars.append(RV_T(w, shape=T_shape))
    else:
      random_vars.append(RV_T(w, shape=T_shape))

    num_experts = D.shape[-1]
    p = RV_p( alpha_p, beta_p, shape=(num_experts,))
    random_vars.append(p)
    q = RV_q( alpha_q, beta_q, shape=(num_experts,))
    random_vars.append(q)

    self.sampler = GibbsSampler(random_vars, data={'D':D}, seed=seed)


  def get_ground_truth(self, sample):
    return sample.T.mean(axis=self.T_shape[0:-2])

  def sample(self, draws,  burn_in=0, chains=1):
    self.sample = self.sampler.sample(draws, burn_in, chains)
    return self.sample
