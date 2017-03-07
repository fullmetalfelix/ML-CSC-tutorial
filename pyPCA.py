import numpy as np
import math
import random

def covariance (X): # function for matrix X
    [nX,mX]=X.shape # get the size of the matrix X
    meanX = np.divide(np.sum(X, axis=0),nX)  # mean row of matrix X
    zX = X - np.kron(np.ones((nX,1)),meanX) # zX = [X - meanX]
    covX = np.divide(np.dot(zX.T,zX),nX-1) # covariance matrix
    return covX

def PCA(X):
    XCov=covariance(X) # this is the same as XCov=np.cov(Xn.T)
    D, V = np.linalg.eig(XCov) # D is eigval and V is eigvec
    Yn=np.dot(X,V)             # perform the linear transformation
                               # by multiplying the original matrix
                               # with eigenvector
    return V,Yn,D
