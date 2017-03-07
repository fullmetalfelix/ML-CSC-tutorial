
import numpy as np

def Kmeans(X,K):

    [nbData,data_dim]= X.shape # get the size of data

    # init the centroids randomly
    data_min=np.amin(X,axis=0)  # Minima along the first axis
    data_max=np.amax(X,axis=0)  # Maxima along the first axis
    data_diff = data_max - data_min

    # every row is a centroid (mu_k)
    mu_k= np.ones((K, data_dim))*np.random.rand(K,data_dim)
    for i in range(1,len(mu_k)+1):
        mu_k[i-1,:]=mu_k[i-1,:] * data_diff # useful if the data is not normalized
        mu_k[i-1,:]=mu_k[i-1,:] + data_min # to put ahead of data_min
    # end mu_k initiation

    # number stopping at start
    pos_diff =1.0

    # main loop until
    while pos_diff > 0.0:
        # E-step
        assignment=[]#None

        # assign each data point to the closest mu_k (centroid)
        for n in range(1,len(X[:,1])+1):

            # assign each data point for cluster 1
            min_diff = (X[n-1,:] - mu_k[0,:]) # the sum squared clustering func (line 1)
            min_diff = np.dot(min_diff,min_diff.T)   # the sum squared clustering func (line 2)
            curAssignment =1

            # assign each data point for cluster k
            for k in range(2,K+1):
                diff2c =  X[n-1,:] - mu_k[k-1,:]
                diff2c =  np.dot(diff2c,diff2c.T)

                # compare the distance of diff to assign in which cluster the data point belong to
                if min_diff >= diff2c:
                    curAssignment = k
                    min_diff = diff2c

            # assign the n-th dataPoint
            assignment = np.append(assignment,curAssignment)

        # for the stopping criterion
        oldPositions=mu_k

        # M-step
        # recalculate the positions of the centroids
        mu_k = np.zeros((K,data_dim))
        pointsInCluster = np.zeros((K,1))

        for n in range(1,len(assignment)+1):

            nint=int(assignment[n-1]) # convert n to be int type
            # sum all of data points in each cluster
            mu_k[nint-1,:] = mu_k [nint-1,:]  + X[n-1,:]

            # count how many data in each cluster
            pointsInCluster[nint-1,0] = pointsInCluster[nint-1,0] + 1

        for k in range(1,K+1):
            print "k= \n %s" %k
            if pointsInCluster[k-1,0] != 0:
                # take the mean of all data point per each cluster
                mu_k[k-1,:]= np.divide(mu_k[k-1,:],pointsInCluster[k-1,0])

            else:
                # set cluster randomly to new position
                mu_k[k-1,:] =  (np.random.rand(1,data_dim) * data_diff) + data_min

        pos_diff_temp = sum((mu_k - oldPositions)**2)
        pos_diff=sum(pos_diff_temp)
        print "mu_k (centroid) values (at the end): \n %s" % mu_k # print the centroids

    assignment1=assignment.astype(int) # to convert the cluster index to be integer
    return assignment1, mu_k
