import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

def KNN(traindata, testdata, K):

    #Find distance with all training datapoints, sort and poll
    dist = np.zeros((np.size(traindata,0),2)) # we initiate dist matrix, 
                                          # the first col=Euclidean dist, the second col=classes
    expclass = np.zeros((np.size(testdata,0),1)) # we initiate expclass (this is the class estimation)

    for i in range(0, np.size(testdata,axis=0)): 
        # we loop for every test data and calculate its distance to all training data
        x=testdata[i,:]
        x=x.reshape((np.size(x,axis=0), 1))
        x=np.transpose(x)

        for j in range(1, np.size(traindata,axis=1)): # we loop for every features (column)
            # j starts with 1, because we store the input data from col 2
            # we accummulate the distance for all features (column)
            dist[:, 0] =  dist[:, 0] + (traindata[:, j] - x[0,j]) ** 2

        dist[:,0] = np.sqrt(dist[:,0]) # Euclidean distance takes a root
        classes = traindata[:,0]       # all classes of training data
        dist[:,1]=classes              # Store the classes in the dist
    
        I = np.argsort(dist[:,0]) # we sort the dist on the first column (the real Euclidean dist)
                                  # it returns the index starting from the closest dist
    
        poll=dist[I,:] # we use the above index, poll is the sorted array (based on Euclidean dist)

        if np.mod(K,2)==1: # it means modulus = 1, if K/2. --> K=odd
            temp0=poll[0 : K, 1] # we select only K closest data  
            temp0=temp0.reshape((np.size(temp0,axis=0), 1))
            [m1,c1]=stats.mode(temp0, axis=0) # m1 is an array of the most common value in temp0.
            expclass[i,:] = m1
        else: # it means --> K=even
            temp0=poll[0 : K-1, 1] # we select only K-1 closest data (so the mode must be odd)
            temp0=temp0.reshape((np.size(temp0,axis=0), 1))
            [m1,c1]=stats.mode(temp0, axis=0) # m1 is an array of the most common value in temp0.
            expclass[i,:] = m1

    CM=confusion_matrix(testdata[:,0], expclass) # compare testdata VS expclass (estimated classes of KNN)
    return expclass, CM
