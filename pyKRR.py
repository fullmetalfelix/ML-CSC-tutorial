import numpy




class KRRsolver:
    
    
    def __init__(self):
        
        self.kmatrix = None
        self.Dmatrix = None
        
        self.alpha = 0.5
        
        self.nTrain = 0
        self.trainIn = None
        self.trainOut = None
        
        print("KRR solver initialised.")
        
    
    
    def Train(self, trainIn, trainOut):
        
        self.nTrain = trainIn.shape[0]
        self.trainIn = trainIn
        self.trainOut = trainOut
        
        # start with a 0 matrix of the right size
        self.kmatrix = numpy.zeros((self.nTrain,self.nTrain))

        for i in range(self.nTrain):
            for j in range(0,i+1):

                # compute the distance between molecule i and j
                self.kmatrix[i,j] = numpy.linalg.norm(trainIn[i]-trainIn[j])

                # it is the same between j and i!
                self.kmatrix[j,i] = self.kmatrix[i,j]

            # add the -alpha on the diagonal
            self.kmatrix[i,i] -= self.alpha

        # invert the matrix
        self.kmatrix = numpy.linalg.inv(self.kmatrix)
        print("Training completed.");
    
    
    def Evaluate(self, validM):
        
        nValid = validM.shape[0]
        
        self.Dmatrix = numpy.zeros((nValid, self.nTrain))
        for i in range(nValid):
            for j in range(self.nTrain):
                self.Dmatrix[i,j] = numpy.linalg.norm(validM[i]-self.trainIn[j])
        
        Epredict = numpy.dot(self.trainOut, numpy.dot(self.kmatrix, numpy.transpose(self.Dmatrix)))
        
        return Epredict
    
    
    
                
                
                
                
                
                
