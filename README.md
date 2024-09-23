# Bayesian STAPLE
An algorithm to estimate the ground truth and performance parameter from a set of raters' segmentations.

## Installation

```
pip install bstaple
```

## Example of usage

```
import numpy as np 
from bstaple import BayesianSTAPLE

rater1 = [0,0,0,1,1,1,0,0,0,0,0]
rater2 = [0,0,0,0,1,1,1,0,0,0,0]
rater3 = [0,0,0,0,1,1,1,0,0,0,0]
D = np.stack([rater1, rater2, rater3], axis=-1)

bayesianSTAPLE = BayesianSTAPLE(D)
trace = bayesianSTAPLE.sample(draws=10000, burn_in=1000, chains=3)
```
Extract the estimated ground truth:
```
soft_ground_truth = trace.T.mean(axis=(0,1)).values
```
Plot the raters' sensitivities and specifities:
```
import arviz as az
ax = az.plot_forest(
    trace,
    var_names=["p", "q"],
    hdi_prob=0.95,
    combined=True
  ) 
```

## Arguments
- __D: array of {0,1} elements__   
    Raters' labels. This array must have this shape:  
    ( dim_1, dim_2, ..., dim_N, raters).  
    The first N dimensions refer to the data labeled by the raters.    
    If repeated_labeling=True the shape must be:  
    (dim_1, dim_2, ..., dim_N, iterations, raters).  
- __w: 'hierarchical', [0,1] or array of [0,1] elements, default='hierarchical'__    
    If it is "hierarchical", this probability will be considered as a random variable and it will be estimated from the sampling.  
    If it is a value between 0 and 1, all the items of the ground truth will have the same probability.  
    If it is an array, each item of the ground truth will have the probability specified by the array. In this case, the w-array must have shape ( dim_1, dim_2, ..., dim_N).  
- __repeated_labeling: boolean, default=False__:  
    Set to 'True' if the raters have made labeled multiple times for the same input. In this case, the data has to have shape (dim_1, dim_2, ..., dim_N, iterations, raters). 
- __alpha_p: int, array of int, optional__:  
    Number of true positives.  
- __beta_p: int, array of int, optional__:  
    Number of false positives.  
- __alpha_q: int, array of int, optional__:  
    Number of true negatives.  
- __beta_q: int, array of int, optional__:  
    Number of false negatives.  
- __alpha_w: int, array of int, optional__:  
    Number of labels 1 that are expected to be in the ground truth.  
- __beta_w: int, array of int, optional__:  
    Number of labels 0 that are expected to be in the ground truth.  
- __seed: int, array of int, optional__:  
    Seed for the sampling algorithm.  



 

