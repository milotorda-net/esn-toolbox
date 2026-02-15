from numpy import *
from numpy import linalg, corrcoef
from scipy import special
from sklearn.neighbors import KDTree
    
def TE_search_space_bootstrapp(source, target, kHistory, kTau, lHistory, lTau, u, k, n_permutations=2):
    #search space preparation
    X=matrix(target[max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau):len(target)]).T
    Y=zeros((len(target)-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),lHistory))
    Z=zeros((len(target)-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),kHistory))
    
    for i in range(max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),len(target)):
        Y[i-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),:]=source[i-(u+lHistory*lTau-lTau):i-(u-1)][((u+lHistory*lTau-lTau)%lTau+(lTau-u))%lTau::lTau]
        Z[i-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),:]=target[i-(1+kHistory*kTau-kTau):i][((1+kHistory*kTau-kTau)%kTau+(kTau-1))%kTau::kTau]
    
    
    Y_bootstrapped=zeros((n_permutations+1,len(target)-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),lHistory))
    Y_bootstrapped[0,:,:]=Y
    for i in range (n_permutations):
        permutations=random.permutation((shape(Y)[0]))
        Y_bootstrapped[i+1,:,:]=Y[permutations]
        
    return X, Y_bootstrapped, Z

def TE_calculation (X, Y, Z, kHistory, kTau, lHistory, lTau, u, k):
    
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
     
    return mean(sumDigammaz-sumDigammayz-sumDigammaxz) + special.digamma(k)


def significance_TE(source, target, kHistory, kTau, lHistory, lTau, u, k, n_permutations):
    temp=TE_search_space_bootstrapp(source, target, kHistory, kTau, lHistory, lTau, u, k, n_permutations)
    distributions=zeros(n_permutations+1)
    p_value=0
    for i in range(n_permutations+1):
        distributions[i]=TE_calculation(temp[0],temp[1][i,:,:],temp[2],kHistory, kTau, lHistory, lTau, u, k)
        if (i>=1):
            if(distributions[i]>distributions[0]):
                p_value+=1
    
    return p_value/n_permutations,distributions[0]

            
def main():
    print("TE permutation test")

if __name__ == '__main__':
    main()    



    
