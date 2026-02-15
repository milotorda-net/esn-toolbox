from numpy import *
from scipy import special
from sklearn.neighbors import KDTree
import numba as nb


@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.float64[:,:,:],nb.float64[:,:])) (nb.float64[:],nb.float64[:],nb.int64,nb.int64,nb.int64,nb.int64,nb.int64,nb.int64,nb.int64),nopython=True)
def TE_search_space_bootstrapp(source, target, kHistory, kTau, lHistory, lTau, u, k, n_permutations):
    
	"""
	Search space preparation for transfer entropy calculations and permutation tests
    	"""
    upper=len(target)
    lower=max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau)
    X=empty((upper-lower, 1))
    X[:,0]=target[lower:upper]
    Y=zeros((upper-lower,lHistory))
    Z=zeros((upper-lower,kHistory))
    
    for i in range(lower,upper):
        Y[i-lower,:]=source[i-(u+lHistory*lTau-lTau):i-(u-1)][((u+lHistory*lTau-lTau)%lTau+(lTau-u))%lTau::lTau]
        Z[i-lower,:]=target[i-(1+kHistory*kTau-kTau):i][((1+kHistory*kTau-kTau)%kTau+(kTau-1))%kTau::kTau]
    
    Y_bootstrapped=zeros((n_permutations+1,upper-lower,lHistory))
    Y_bootstrapped[0,:,:]=Y
    permutations=arange((upper-lower))
    for i in range (n_permutations):
        random.shuffle(permutations)
        Y_bootstrapped[i+1,:,:]=Y[permutations]
       
    return X, Y_bootstrapped, Z

@nb.jit(nb.float64(nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.int64,nb.int64,nb.int64,nb.int64,nb.int64,nb.int64))
def TE_calculation (X, Y, Z, kHistory, kTau, lHistory, lTau, u, k):
    
	"""
	Computes transfer entropy value.
	"""

    XYZ=hstack((X,Y,Z))
    XZ=XYZ[:,r_[0,lHistory+1:lHistory+1+kHistory]]
    YZ=XYZ[:,r_[1:lHistory+1,lHistory+1:lHistory+1+kHistory]]
    Z=XYZ[:,r_[lHistory+1:lHistory+1+kHistory]]
    
    #k-NN search
    treexyz = KDTree(XYZ, metric="chebyshev")
    treexz = KDTree(XZ, metric="chebyshev")
    treeyz = KDTree(YZ, metric="chebyshev")
    treez = KDTree(Z, metric="chebyshev")
    
    epsilon=nextafter(treexyz.query(XYZ, k=k+1)[0][:,k],0)
    sumxz=treexz.query_radius(XZ, r=epsilon, count_only=True)
    sumyz=treeyz.query_radius(YZ, r=epsilon, count_only=True)
    sumz=treez.query_radius(Z, r=epsilon, count_only=True)
               
    sumDigammaxz=special.digamma(sumxz)
    sumDigammayz=special.digamma(sumyz)
    sumDigammaz=special.digamma(sumz)
     
    return (mean(sumDigammaz-sumDigammayz-sumDigammaxz) + special.digamma(k))

#@nb.jit(nb.types.Tuple((nb.float64,nb.float64))(nb.float64[:],nb.float64[:],nb.int64,nb.int64,nb.int64,nb.int64,nb.int64,nb.int64,nb.int64))
def significance_TE(source, target, kHistory, kTau, lHistory, lTau, u, k, n_permutations):
    
	"""
	Computes p-value and transfer enropy value.
	"""

    def TE_calc(Y):
        return TE_calculation(temp[0],Y,temp[2],kHistory, kTau, lHistory, lTau, u, k)
        
    temp=TE_search_space_bootstrapp(source, target, kHistory, kTau, lHistory, lTau, u, k, n_permutations)
    distributions=zeros(n_permutations)
    distribution=TE_calculation(temp[0],temp[1][0,:,:],temp[2],kHistory, kTau, lHistory, lTau, u, k)
    
    distributions=list(map(TE_calc, temp[1]))
        
    p_value=sum(distributions>distribution)

    return (p_value/n_permutations,distribution)

#@nb.jit(nb.float64[:,:,:](nb.float64[:,:],nb.float64[:],nb.int64,nb.int64,nb.int64,nb.int64,nb.int64,nb.int64,nb.int64))
def pairwise_TE (X,kHistory, kTau, lHistory, lTau, u, k, n_permutations,significance=True):
	"""
	Calculates Transfer entropy matrix of the whole system.	
	
	Args:
		X: ndarray
			N x K time series matrix of the whole system. N number of nodes in the system and K is the length of the coresponding time series.
		kHistory: int 
			Target history length.
		kTau: int
			Target delay length.
		lHistory: int 
			Source history length.
		lTau: int
			Source delay length
		u: int
			Source-target lag
		k: int
			Number of nearest neighbours in the KSG estimator.
		n_permutations: int 
			Number of source permutations to compute for the permutation test.
		significance: bool
			If True performs also the permutation test 
				else only computes transfer entropy matrix.

	Returns: ndarrays
			Two N x N matrices. The first is transfer entropy between each node of the system. The second is coresponding p-value.
	"""
    def significance_test(S,T):
        return significance_TE(S,T,kHistory, kTau, lHistory, lTau, u, k, n_permutations)
    
    def TE_calc(Y,X):
        temp=TE_search_space_bootstrapp(Y, X, kHistory, kTau, lHistory, lTau, u, k, n_permutations=0)
        return TE_calculation(temp[0],temp[1][0,:,:],temp[2],kHistory, kTau, lHistory, lTau, u, k)
    
    resize=X.shape[0]
    TE=zeros((resize,resize,2))
    
    source=repeat(X,resize,axis=0)
    target=tile(X,(resize,1))
    if significance:
        TE=list(map(significance_test,source,target))
        return reshape(TE,(resize,resize,2))
    else:
        TE=list(map(TE_calc,source,target))
        return reshape(TE,(resize,resize))

def main():
    print("TE permutation test")

if __name__ == '__main__':
    main()    
