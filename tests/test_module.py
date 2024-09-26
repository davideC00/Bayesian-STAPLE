import numpy as np

def test_module():
    from bstaple import BayesianSTAPLE
    rater1 = [0,0,0,1,1,1,0,0,0,0,0]
    rater2 = [0,0,0,0,1,1,1,0,0,0,0]
    rater3 = [0,0,0,0,1,1,1,0,0,0,0]
    D = np.stack([rater1, rater2, rater3], axis=-1)

    bayesianSTAPLE = BayesianSTAPLE(D)
    trace = bayesianSTAPLE.sample(draws=20, burn_in=10, chains=2)
    


if __name__ == "__main__":
    test_module()
    print("Everything passed")