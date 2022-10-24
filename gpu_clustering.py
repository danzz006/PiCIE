from sqlite3 import Time
import rmm
import numpy as np
import cupy as cp
import cuml.dask
from cuml.cluster import DBSCAN
from distributed import Client
from dask_cuda import LocalCUDACluster
from time import time
   
    
def getData(samples):
    fset = cp.asarray(np.random.randn(samples,128))
    return fset

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
    cluster = LocalCUDACluster(
                CUDA_VISIBLE_DEVICES=[0,1,2, 3],
            )
    client = Client(cluster)

    rmm.rmm.reinitialize(
    devices=[2]
    )
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

#from cuml.cluster import DBSCAN as dbscan

    # distDbscan = cuml.dask.cluster.DBSCAN(client=client, verbose=False,output_type="cupy", max_mbytes_per_batch=8000, min_samples=200)

    with  cp.cuda.Device(2):
        fset = getData(1500000)
        Dbscan = DBSCAN(min_samples=5)




