import numpy as np
from bstaple import BayesianSTAPLE

def test_readme_example():
    rater1 = [0,0,0,1,1,1,0,0,0,0,0]
    rater2 = [0,0,0,0,1,1,1,0,0,0,0]
    rater3 = [0,0,0,0,1,1,1,0,0,0,0]
    D = np.stack([rater1, rater2, rater3], axis=-1)

    bayesianSTAPLE = BayesianSTAPLE(D)
    trace = bayesianSTAPLE.sample(draws=20, burn_in=10, chains=3)
    soft_ground_truth = bayesianSTAPLE.get_ground_truth(trace)
    assert soft_ground_truth.shape == D.shape[0:-1]

def test_repeated_labeling():
    rater1_first_labelling = [0,0,0,1,1,1,0,0,0,0,0]
    rater1_second_labelling = [0,0,0,1,1,1,0,0,0,0,0]
    rater_1 = np.stack([rater1_first_labelling, rater1_second_labelling], axis=-1)
    rater2_first_labelling = [0,0,0,0,1,1,1,0,0,0,0]
    rater2_second_labelling = [0,0,0,1,1,1,1,0,0,0,0]
    rater_2 = np.stack([rater2_first_labelling, rater2_second_labelling], axis=-1)
    D = np.stack([rater_1, rater_2], axis=-1)

    bayesianSTAPLE = BayesianSTAPLE(D, repeated_labeling=True)
    trace = bayesianSTAPLE.sample(draws=1000, burn_in=100, chains=3)
    soft_ground_truth = bayesianSTAPLE.get_ground_truth(trace)
    assert soft_ground_truth.shape == rater_1.shape[0:-1]


if __name__ == "__main__":
    test_readme_example()
    test_repeated_labeling()
    print("Everything passed")