import numpy as np

def Normaization0(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def Normaization1(X):
    return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def Z_score(X):
    return (X - X.mean(axis=0)) / np.std(X)

