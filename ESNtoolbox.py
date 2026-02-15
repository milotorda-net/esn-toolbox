from numpy import *
from numpy import linalg, corrcoef
from scipy import special
from sklearn.neighbors import KDTree

def AIS (target, kHistory, kTau, k):
	
	"""
	Active information storage computation.
	"""

    #search space preparation
    X=matrix(target[(1+kHistory*kTau-kTau):len(target)]).T
    Z=zeros((len(target)-(1+kHistory*kTau-kTau),kHistory))
    
     
    for i in range((1+kHistory*kTau-kTau),len(target)):
        Z[i-(1+kHistory*kTau-kTau),:]=target[i-(1+kHistory*kTau-kTau):i][((1+kHistory*kTau-kTau)%kTau+(kTau-1))%kTau::kTau]
    
    Z=matrix(Z)
    XZ=hstack((X,Z))
    
    #k-NN search
    treexz = KDTree(XZ, metric="chebyshev")
    treex = KDTree(X, metric="chebyshev")
    treez = KDTree(Z, metric="chebyshev")
        
    kNNxz=treexz.query(XZ, k=k+1)[1]
    temp=X[kNNxz][:,:,0]
    epsylonx=amax(abs(temp[:,0]-temp[:,r_[1:k+1]]),axis=1)
    temp=Z[kNNxz]
    epsylonz=amax(linalg.norm(swapaxes(asarray(temp[:,r_[1:k+1],:]),0,1)-asarray(temp[:,0,:]),inf,axis=2).T,axis=1)
    sumx=treex.query_radius(X, r=squeeze(asarray(epsylonx)), count_only=True)
    sumz=treez.query_radius(Z, r=squeeze(asarray(epsylonz)), count_only=True)
                   
    sumDigammax=special.digamma(sumx-1)
    sumDigammaz=special.digamma(sumz-1)
            
    return mean(-sumDigammaz-sumDigammax) -1/k + special.digamma(k) + special.digamma(len(sumDigammax))

def TE (source, target, kHistory, kTau, lHistory, lTau, u, k):

	"""
	Transfer entropy computation.
	"""

    #search space preparation
    X=matrix(target[max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau):len(target)]).T
    Y=zeros((len(target)-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),lHistory))
    Z=zeros((len(target)-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),kHistory))
    
    for i in range(max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),len(target)):
        Y[i-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),:]=source[i-(u+lHistory*lTau-lTau):i-(u-1)][((u+lHistory*lTau-lTau)%lTau+(lTau-u))%lTau::lTau]
        Z[i-max(u+lHistory*lTau-lTau,1+kHistory*kTau-kTau),:]=target[i-(1+kHistory*kTau-kTau):i][((1+kHistory*kTau-kTau)%kTau+(kTau-1))%kTau::kTau]
    
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

def MC(Win, W,data):

	"""
	Computes memory capacity of the network.
	"""

    resize=Win.shape[0]
    insize=1
    outsize=1
    initLen=100
    trainLen=1000
    testLen=2000
    memory=120
    a=-1
    b=1
    
    #u=random.rand(memory+trainLen+testLen+1,1)*(b-a)+a
    u=data
    Wout=zeros((outsize,resize))
    x=zeros(resize)
    Xtrain=zeros((resize,trainLen))
    Xtest=zeros((resize,testLen))
    Ytest=zeros((memory,testLen))
    Ytrain=zeros((memory,trainLen))
        
    for t in range(memory+trainLen+1):    
        x=tanh(dot(W,x) +dot(Win,squeeze(u[t])))
            
        if(t>=memory+1):            
            Xtrain[:,t-memory-1]=x
        
    for t in  range(memory):   
        Ytrain[t,:]= squeeze(u[memory-t:memory+trainLen-t])
                   
    Wout=dot(Ytrain,linalg.pinv(Xtrain))
    Ytest=zeros((memory,testLen))           
       
    for t in range (testLen):   
        x=tanh(dot(W,x) +dot(Win,squeeze(u[memory+trainLen+1+t])))
        Xtest[:,t-memory-1]=x
        Ytest[:,t]=dot(Wout,x)
        
    MC=zeros(memory)
    for i in range(memory):
        MC[i]=power(corrcoef(Ytest[i,:],squeeze(u[memory+trainLen-i:memory+trainLen+testLen-i]))[0,1],2)         
        
    return sum(MC), Xtest

def LE(W, WI, data, gamma0=10**-12, iterations=2000):

	"""
	Computes Lyapunov exponent of the ESN.
	"""

    q=size(WI)
    lambdasum = 0
    for noda in range(q):
        X = zeros(q)
        X2 = zeros(q)
        X2[noda] += gamma0
        lambdas = zeros(1000)
        for it in range(iterations):
            X = tanh(dot(W, X) + dot(WI, data[it]))
            X2 = tanh(dot(W, X2) + dot(WI, data[it]))
            if (it>=1000):
                difr = X2 - X
                gammaK = sqrt(difr.dot(difr))
                X2 = X + difr * (gamma0 / gammaK)
                lambdas[it-1000] = log(gammaK / gamma0)
                    
        lambdasum += average(lambdas)
        
    return lambdasum / q

def locally_const_predictor(target, kHistory, kTau, k):

	"""
	Computes locally constant prediction error for Ragwitz-Kantz embedding criterion.
	"""

    #search space preparation
    Z=zeros((len(target)-(1+kHistory*kTau-kTau),kHistory))
    X=target[(1+kHistory*kTau-kTau):len(target)]
           
    for i in range((1+kHistory*kTau-kTau),len(target)):
        Z[i-(1+kHistory*kTau-kTau),:]=target[i-(1+kHistory*kTau-kTau):i][((1+kHistory*kTau-kTau)%kTau+(kTau-1))%kTau::kTau]
        
    Z=matrix(Z)
        
    #k-NN search
    treez = KDTree(Z, metric="chebyshev")
            
    kNNz=treez.query(Z, k=k+1)[1]
    prediction=sum(X[kNNz[:,1:(k+1)]],axis=1)/k
    error=mean((prediction-X)**2)
    
    return error
    
def ragwitz(target,kHistoryMax,kTauMax,k):

	"""
	Ragwitz-Kantz embedding criterion grid search. 
	"""

    kHistory=1
    kTau=1
    kHistoryBest=1
    kTauBest=1
    bestPredictionError=locally_const_predictor(target,kHistory,kTau,k)
    
    for i in range(kHistoryMax-1):
        kHistory+=1
        for j in range(kTauMax):
            kTau=j+1
            predictionError=locally_const_predictor(target,kHistory,kTau,k)
            if(predictionError<bestPredictionError):
                bestPredictionError=predictionError
                kHistoryBest=kHistory
                kTauBest=kTau

    return  kHistoryBest, kTauBest  

def entropy(data,k):

	"""
	Computes differential entropy using Kozachenko-Leonenko entropy estimation.
	"""

    #k-NN search
    X=matrix(data).T
    treex = KDTree(X, metric="chebyshev")       
    kNNx=treex.query(X, k=k+1)[0][:,k]    
       
    return mean(log(2*kNNx)) - special.digamma(k) + special.digamma(len(kNNx))

def main():
    print("ESNtoolbox")

if __name__ == '__main__':
    main()