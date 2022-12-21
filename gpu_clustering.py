from sqlite3 import Time
import rmm
import numpy as np
import cupy as cp
from sklearn.datasets import make_blobs
import cuml.dask
from cuml.cluster import DBSCAN
from distributed import Client
from dask_cuda import LocalCUDACluster
import torch
from sklearn.model_selection import ParameterGrid

from time import time
import matplotlib.pyplot as plt   
    
def getData(samples):
    # x = make_blobs(n_samples=samples, centers=27, n_features=3,
    #                random_state=0)

    x = make_blobs(n_samples=samples, centers=27, cluster_std=0.2, n_features=128,
                   random_state=0)

    fset = cp.asarray(x[0])
    # fset = cp.asarray(np.random.normal(0, 0.1,(samples,128)))
    return fset, x[1]

def singleGpuClustering(fs):
    start = time()
    Dbscan.fit(fs)
    end = time()
    print(f"Execution time: {(end-start) * 10**3}ms")
    

    
def multiGpuClustering(fs):
    start = time()
    # distDbscan.fit(fs)
    end = time()
    print(f"Execution time: {(end-start) * 10**3}ms")



if __name__ == "__main__":
    # cluster = LocalCUDACluster(
    #             CUDA_VISIBLE_DEVICES=[0,1,2, 3],
    #         )
    # client = Client(cluster)

    losses = []
    param_grid = {
        'min_samples': [i for i in range(5, 20)], 
        'eps': [0.009, 0.5, 0.7, 0.8, 0.88, 0.9],
        'metric': ['cosine', 'euclidean'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

    params_list = []
    
    fset, labs = getData(50000)
    labs_ = torch.tensor(labs, requires_grad=False)

    for grid in ParameterGrid(param_grid):
        loss = torch.nn.CrossEntropyLoss().cuda()
        loss.requires_grad_ = False
        
        clus = DBSCAN(
                min_samples=grid['min_samples'],
                eps=grid['eps'],
                metric=grid['metric']
            )

        clus.fit(fset)

        preds = clus.labels_
        preds = cp.asnumpy(preds)
        preds = np.array(preds.tolist())
        preds_list = preds.tolist()
        for i in np.unique(labs):
            for x in preds[labs == i]:
                preds_list[x] = [0 if j != i else 1 for j in np.unique(labs)]
            
            
        preds_ = torch.tensor(preds, requires_grad=False)
        l = loss(preds_.float().unsqueeze(dim=1), labs_[labs == i])
        losses.append(l.numpy().item())
        params_list.append(grid)
    print(l)
            

    # ax = plt.axes(projection='3d')
    # rmm.rmm.reinitialize(
    # devices=[2]
    # )
    # cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

#from cuml.cluster import DBSCAN as dbscan

    # distDbscan = cuml.dask.cluster.DBSCAN(client=client, verbose=False,output_type="cupy", max_mbytes_per_batch=8000, min_samples=200)

    # with  cp.cuda.Device(2):
    # l = clus.labels_
    # print(l)
    # print(clus.core_sample_indices_)
    # print(clus.components_)
    # print()
    # Dbscan = DBSCAN(min_samples=5)




