# Bayesian STAPLE
An algorithm that merges raters' labelings and estimates a ground truth and the performance parameters of each rater.  

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
soft_ground_truth = bayesianSTAPLE.get_ground_truth(trace)
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
For other exaples, check the paper of the library: [Link](https://ojs.unito.it/index.php/JAS/article/view/11095).

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


## Testing the library

Point to the directory and run in the shell:
```
poetry install
poetry run python ./tests/test_module.py
```


## Cite 
If you use the library please cite our papers:
```
@article{Mencar_Cazzorla_2025,
    title={Bayes-STAPLE: a python module for Bayesian label fusion},
    volume={2}, url={https://ojs.unito.it/index.php/JAS/article/view/11095},
    DOI={10.13135/3103-1935/11095},
    number={1},
    journal={Journal of Approximation Software},
    author={Mencar, Corrado and Cazzorla, Davide},
    year={2025},
    month={Mar.}
}



@inproceedings{moser_uncertainty_2024,
	location = {Cham},
	title = {Uncertainty Estimation of Raters’ Performance and Ground Truth Through a Bayesian Extension of {STAPLE}},
	volume = {2169},
	isbn = {978-3-031-68301-5 978-3-031-68302-2},
	url = {https://link.springer.com/10.1007/978-3-031-68302-2_8},
	doi = {10.1007/978-3-031-68302-2_8},
	series = {Communications in Computer and Information Science},
	pages = {91--101},
	booktitle = {Database and Expert Systems Applications - {DEXA} 2024 Workshops},
	publisher = {Springer Nature Switzerland},
	author = {Cazzorla, Davide and Mencar, Corrado},
	editor = {Moser, Bernhard and Fischer, Lukas and Mashkoor, Atif and Sametinger, Johannes and Glock, Anna-Christina and Mayr, Michael and Luftensteiner, Sabrina},
	urldate = {2025-08-29},
	date = {2024},
}

```


