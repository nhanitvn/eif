import time
import numpy as np
import eif as iso


def gen_data():
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    Nobjs = 500
    np.random.seed(1)
    x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
    #Add manual outlier
    x[0]=3.3
    y[0]=3.3
    X=np.array([x,y]).T

    return X

def test_sequential_running():
    t1 = time.time()
    X = gen_data()
    
    F0  = iso.iForest(X, ntrees=100, sample_size=256, ExtensionLevel=0)
    t2 = time.time()
    print('Forest Building took %s seconds' % (t2 -t1))
    S0 = F0.compute_paths(X_in=X)
    t3 = time.time()
    print('Anomaly Score Computation took %s seconds' % (t3 -t2))

def test_parallel_running():
    
    t1 = time.time()
    X = gen_data()
    
    F0  = iso.iForest(X, ntrees=100, sample_size=256, ExtensionLevel=0, n_jobs=10)
    t2 = time.time()
    print('Forest Building took %s seconds' % (t2 -t1))
    S0 = F0.compute_paths(X_in=X)
    t3 = time.time()
    print('Anomaly Score Computation took %s seconds' % (t3 -t2))
