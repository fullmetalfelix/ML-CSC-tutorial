import numpy, random


def Evaluation_dummy(element):
    return random.random()

class GAEngine:
    
    
    def __init__(self, popSize, dnaSize, scale=1.0):
        
        self.populationSize = popSize
        self.dnaSize = dnaSize
        self.scale = scale

        self.population = scale*(2*numpy.random.rand(popSize, dnaSize)-1);
        self.generation = 0
        self.best = None
        self.bestFit = None
        
        self.mutationRate = 0.01
        self.mutationScale = 0.1
        
        self.Evaluate = Evaluation_dummy
        self.tau = 0.1
    

    
    def Evolve(self):
        
        # evaluate all elements
        fitness = numpy.zeros(self.populationSize)
        ps = numpy.arange(self.populationSize) / (self.populationSize-1)
        ps = (self.tau) * numpy.exp(-ps*self.tau)
        ps /= numpy.sum(ps)
        
        for i in range(self.populationSize):
            element = self.population[i]
            fitness[i] = self.Evaluate(element)
            
        # sort population by fitness - inverse order
        idx = numpy.argsort(-fitness)
        fitness = fitness[idx]
        self.population = self.population[idx]
        print('generation[{}] fitness: best {}, avg {}, worse {}'.format(self.generation, fitness[0], numpy.mean(fitness), fitness[self.populationSize-1]))
        
        if self.bestFit == None:
            self.best = self.population[0]
            self.bestFit = fitness[0]
        else:
            if self.bestFit < fitness[0]:
                self.best = self.population[0]
                self.bestFit = fitness[0]
            
        
        # pick pairs for reproduction
        idx = numpy.arange(self.populationSize, dtype=numpy.int32)
        mfactor = self.mutationRate/self.dnaSize
        newpop = numpy.zeros((self.populationSize, self.dnaSize))
        
        for i in range(self.populationSize):
            
            parentsIdx = numpy.random.choice(idx, size=2, p=ps)
            parents = self.population[parentsIdx]
            
            # do the mixing
            mixer = numpy.random.choice([0,1], size=self.dnaSize)
            son = parents[0] * mixer + parents[1] * (1-mixer)
            
            # mutate
            mutator = numpy.random.choice([0,1], size=self.dnaSize, p=[1-mfactor, mfactor])
            mutator = (2*numpy.random.rand(self.dnaSize)-1) * self.mutationScale * mutator
            son += mutator
            
            newpop[i] = son
        
        self.population = newpop
        self.generation += 1
        
        return [fitness[0], numpy.mean(fitness), fitness[self.populationSize-1]]
    
        
    def TrySelection(self, n):
        
        ps = numpy.arange(self.populationSize) / (self.populationSize-1)
        ps = (self.tau) * numpy.exp(-ps*self.tau)
        ps /= numpy.sum(ps)
        
        idx = numpy.arange(self.populationSize, dtype=numpy.int32)
        
        selected = numpy.random.choice(idx, size=n, p=ps)
        
        return selected
        
        
        
        
        
        
        
        
        