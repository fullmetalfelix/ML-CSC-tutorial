import numpy
import math
import random
import numpy

def TanSigmoid(x):
    return numpy.tanh(x) #(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def Relu(x):
    if x>0:
        return x
    else:
        return 0

def TanSigmoid_D(x):
    return 1-TanSigmoid(x)*TanSigmoid(x)

def Relu_D(x):
    if x>0:
        return 1
    else:
        return 0

def CalcZ(x,matrix,bias):
    return numpy.dot(matrix,x) + bias
 
def KronProd(x):
    return numpy.kron(x[0],x[1])
 
def Evaluate(x, network):
        return network.Evaluate(x)


class NeuralNetwork(object):
 
 
    def __init__(self, nI, nO, nH):
 
        self.nInputs = nI
        self.nOutputs = nO
        self.nHidden = nH
 
        self.learnRate = 0.95
        self.regularization = 0.0
 
        self.inputs = numpy.zeros(self.nInputs)
 
        self.W12 = 2*numpy.random.rand(self.nHidden,self.nInputs)-1
        self.W23 = 2*numpy.random.rand(self.nOutputs,self.nHidden)-1
        self.B2 = 2*numpy.random.rand(self.nHidden)-1
        self.B3 = 2*numpy.random.rand(self.nOutputs)-1
 
        self.actfun = numpy.vectorize(TanSigmoid, otypes=[numpy.float])
        self.actfun_D = numpy.vectorize(TanSigmoid_D, otypes=[numpy.float])
 
        #self.actfunV = numpy.vectorize(TanSigmoid, otypes=[numpy.float])
 
 
 
    ## Compute the output of the neural network given the input values
    def Evaluate(self, inputs):
 
        z2 = numpy.dot(self.W12, inputs) + self.B2
        a2 = self.actfun(z2)
        z3 = numpy.dot(self.W23, a2) + self.B3
        a3 = self.actfun(z3)
 
        result = a3
        return result
 
 
 
    def GradStep(self, inputList, outputList):
        
        nins = inputList.shape[0]
        
        z2s = numpy.apply_along_axis(CalcZ,1,inputList,self.W12,self.B2)
        a2s = self.actfun(z2s)
 
        z3s = numpy.apply_along_axis(CalcZ,1,a2s,self.W23,self.B3)
        a3s = self.actfun(z3s)
 
        error = 0.5*(a3s - outputList)*(a3s - outputList)
 
        d3 = (a3s - outputList)*(self.actfun_D(z3s))
        d2 = numpy.apply_along_axis(CalcZ,1,d3,numpy.transpose(self.W23),0)
        d2 = d2 * self.actfun_D(z2s)
        
        grad23 = numpy.kron(d3[0],a2s[0]) * 0
        for i in xrange(nins):
            grad23 += numpy.kron(d3[i],a2s[i])
        grad23 = grad23 / nins

        #tr32 = numpy.transpose( numpy.asarray([d3,a2s]) )
        grad12 = numpy.kron(d2[0],inputList[0]) * 0
        for i in xrange(nins):
            grad12 += numpy.kron(d2[i],inputList[i])
        grad12 = grad12 / nins
        
        #tr21 = numpy.transpose( numpy.asarray([d2,inputList]) )
 
        #grad23 = numpy.apply_along_axis(KronProd,1,tr32)
        #grad12 = numpy.apply_along_axis(KronProd,1,tr21)
 
        gradB2 = numpy.mean(d2, axis=0)
        gradB3 = numpy.mean(d3, axis=0)
        #grad23 = numpy.mean(grad23, axis=0)
        grad23 = numpy.reshape(grad23,self.W23.shape)
        #grad12 = numpy.mean(grad12, axis=0)
        grad12 = numpy.reshape(grad12,self.W12.shape)
 
        reg = self.W12 * self.regularization * 2
        self.W12 = self.W12 -grad12 - reg
        reg = self.W23 * self.regularization * 2
        self.W23 += -self.learnRate * grad23 - reg
 
        reg = self.B2 * self.regularization * 2
        self.B2 += -self.learnRate * gradB2 - reg
        reg = self.B3 * self.regularization * 2
        self.B3 += -self.learnRate * gradB3 - reg
 
        return numpy.mean(2*error.flatten())
 
 
    def StochasticGradStep(self,inputList, outputList, batchSize):
 
        indexes = xrange(len(inputList))
        selection = random.sample(indexes, batchSize)
        batchIn = numpy.take(inputList,selection, axis=0)
        batchOut = numpy.take(outputList,selection, axis=0)
 
        error = self.GradStep(batchIn, batchOut)
 
        return error




