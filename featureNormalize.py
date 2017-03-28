import numpy as np

def normalize(featureMat):
    mu = np.mean(featureMat, axis=0)
    sigma = np.std(featureMat, axis=0, ddof=1)
    return (featureMat-mu)/sigma