import numpy as np
import math as m
import random as rn
import tensorflow as tf
import matplotlib.pyplot as plt
import os

####################################################################
#Data
####################################################################
height = 32
width = 32
channels = 3
input_shape = (height, width, channels)
n_classes = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

index, _ = np.where(y_train==0)
images = x_train[index]
labels = y_train[index]
x_val = images[0:500]
y_val = labels[0:500]
x_train2 = images[500:]
y_train2 = labels[500:]

for i in range(1, n_classes):
    index, _ = np.where(y_train==i)
    images = x_train[index]
    labels = y_train[index]
    x_val = np.concatenate((x_val, images[0:500]), axis=0)
    y_val = np.concatenate((y_val, labels[0:500]), axis=0)
    x_train2 = np.concatenate((x_train2, images[500:]), axis=0)
    y_train2 = np.concatenate((y_train2, labels[500:]), axis=0)

permutation = np.random.permutation(5000)
x_val = x_val[permutation]
y_val = y_val[permutation]

permutation = np.random.permutation(45000)
x_train2 = x_train2[permutation]
y_train2 = y_train2[permutation]

x_train = x_train2
y_train = y_train2

####################################################################
#Data space 'elu', 'exponential', 'gelu', 'linear', 'relu', 'selu',
####################################################################
dataActivationFunction = ['elu', 'exponential', 'gelu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']
dataOptimizers = [tf.keras.optimizers.legacy.Adadelta(), 
                tf.keras.optimizers.legacy.Adagrad(), 
                tf.keras.optimizers.legacy.Adam(),
                tf.keras.optimizers.legacy.Adamax(),
                tf.keras.optimizers.legacy.Ftrl(),
                tf.keras.optimizers.legacy.Nadam(),
                tf.keras.optimizers.legacy.RMSprop(),
                tf.keras.optimizers.legacy.SGD()]

dataInitializers = [tf.keras.initializers.GlorotNormal(),
                    tf.keras.initializers.GlorotUniform(),
                    tf.keras.initializers.HeNormal(),
                    tf.keras.initializers.HeUniform(),
                    tf.keras.initializers.LecunNormal(),
                    tf.keras.initializers.LecunUniform(),
                    tf.keras.initializers.RandomNormal(),
                    tf.keras.initializers.RandomUniform()]

####################################################################
#Config
####################################################################
mutationProb = 0.1
paramsDisc = [len(dataOptimizers), len(dataInitializers)]
paramsArchConvDisc = [len(dataActivationFunction)]
paramsArchDenDisc = [len(dataActivationFunction)]
archConv = np.array([32, 120, 6, 2, 3]) #min kernels, max kernels, total layers, min size kernel, max size kernel
archDen = np.array([10, 100, 4]) # min kernels, max kernels, total layers
paramsCont = np.array([
    [15, 20], #Epochs
    [20, 60], #Batch
    [0.001, 0.5] #LR
])
dirMaker = f'/zfs-home/2023/202304070c/topaper/models/models_1/gen'

lossFunction = tf.keras.losses.SparseCategoricalCrossentropy()
#lossFunction = tf.keras.losses.BinaryCrossentropy()
#lossFunction = tf.keras.losses.MeanSquaredError()

class sample:
    def __init__(self):
        global archConv, archDen, paramsArchConvDisc, paramsArchDenDisc, paramsCont, paramsDisc
        self.paramsCont = np.zeros(shape=len(paramsCont)) # epochs, batch, lr
        self.paramsDisc = np.zeros(shape=len(paramsDisc)) # optimizer, initializers
        #self.paramsDiscP = np.random.uniform(0,1,(len(paramsDisc, paramsDisc[0])))
        for i in range(len(paramsCont)):
            self.paramsCont[i] = rn.uniform(paramsCont[i, 0], paramsCont[i, 1])
        for i in range(len(paramsDisc)):
            self.paramsDisc[i] = rn.randint(0, paramsDisc[i]-1)
            
        self.archConv = np.random.randint(low=archConv[0], high=archConv[1], size=(archConv[2]))
        self.archConvKernels = np.random.randint(low=archConv[3], high=archConv[4], size=(archConv[2]))
        self.archConvParamsDisc = np.zeros(shape=(self.archConv.shape[0]))
        for i in range(self.archConv.shape[0]):
            self.archConvParamsDisc[i] = rn.randint(0, paramsArchConvDisc[0]-1)

        self.archDen = np.random.randint(low=archDen[0], high=archDen[1], size=(archDen[2]))
        self.archDenParamsDisc = np.zeros(shape=(self.archDen.shape[0]))
        for i in range(self.archDen.shape[0]):
            self.archDenParamsDisc[i] = rn.randint(0, paramsArchDenDisc[0]-1)

    def setSample(self, paramsCont, paramsDisc, archConv, archConvK, archDen, paramsArchConvDisc, paramsArchDenDisc):
        self.paramsCont = paramsCont
        self.paramsDisc = paramsDisc
        self.archConv = archConv 
        self.archConvKernels = archConvK
        self.archDen = archDen
        self.archConvParamsDisc = paramsArchConvDisc
        self.archDenParamsDisc = paramsArchDenDisc
    
    def setArch(self, archConv, archConvK, archDen):
        self.archConv = archConv
        self.archConvKernels = archConvK
        self.archDen = archDen
    
    def getArch(self):
        return self.archConv, self.archConvKernels, self.archConvParamsDisc, self.archDen, self.archDenParamsDisc 

    def getParams(self):
        return self.paramsCont, self.paramsDisc

    def buildModel(self):
        global archConv, archDen
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(
            int(self.archConv[0]), (int(self.archConvKernels[0]), int(self.archConvKernels[0])), 
            activation=dataActivationFunction[int(self.archConvParamsDisc[0])], 
            input_shape=(32, 32, 3),
            kernel_initializer = dataInitializers[int(self.paramsDisc[1])]))
        model.add(tf.keras.layers.MaxPool2D())
        model.add(tf.keras.layers.BatchNormalization())
        cont = 0
        for i in range(1, int(archConv[2])):
            model.add(tf.keras.layers.Conv2D(
                int(self.archConv[i]), (int(self.archConvKernels[i]), int(self.archConvKernels[i])), 
                activation = dataActivationFunction[int(self.archConvParamsDisc[i])], 
                kernel_initializer = dataInitializers[int(self.paramsDisc[1])]))
            if cont == 2: 
                model.add(tf.keras.layers.MaxPool2D())
                model.add(tf.keras.layers.BatchNormalization())
                cont = 0
            else:
                cont += 1    

        model.add(tf.keras.layers.Flatten())
        for i in range(int(archDen[2])):
            model.add(tf.keras.layers.Dense(
                int(self.archDen[i]), activation=dataActivationFunction[int(self.archDenParamsDisc[i])],
                kernel_initializer=dataInitializers[int(self.archDenParamsDisc[i])]))
            model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
            
        return model

    def trainModel(self, model, verbose):
        global dataOptimizers, lossFunction
        optimizer = dataOptimizers[int(self.paramsDisc[0])]
        optimizer.lr = self.paramsCont[2]
        model.compile(optimizer=optimizer,
                loss=lossFunction,
                metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=int(self.paramsCont[0]), 
                            validation_data=(x_val, y_val), 
                            batch_size=int(self.paramsCont[1]), verbose=verbose)
    
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=2)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=verbose)

        return history, test_loss, test_acc, val_loss, val_acc, model

    def fitnessModel(self, verbose):
        model = self.buildModel()
        model.summary()
        history, loss_test, score_test, loss_val, score_val, model = self.trainModel(model, verbose)
        return model, score_test, score_val, history

    def show(self):
        print("Params Cont:", self.paramsCont)
        print("Params Disc:", self.paramsDisc)
        print("Arch Conv:", self.archConv)
        print("Arch Conv Kernels:", self.archConvKernels)
        print("Arch Conv Params:", self.archConvParamsDisc)
        print("Arch Den:", self.archDen)
        print("Arch Den Params:", self.archDenParamsDisc)

    def save(self, gen, sampleN, popType, params, score_test, score_val):
        totalParams = params
        with open(f'{dirMaker}{gen}/model{sampleN}:popType_{popType},it_{gen},score_{score_val:.4f}.txt', 'w') as f:
                f.write(f'Params Cat\n')
                for e in self.paramsDisc:
                    f.write(f"{e}\n")
                f.write(f'Params Cont\n')
                for e in self.paramsCont:
                    f.write(f"{e}\n")
                f.write(f'Architecture Conv - Kernels\n')
                for e in self.archConv:
                    f.write(f"{int(e)},")
                f.write(f'\n')
                f.write(f'Architecture Conv - Kernel size\n')
                for e in self.archConvKernels:
                    f.write(f"{int(e)},")
                f.write(f'\n')
                f.write(f'Architecture Conv - Activation function\n')
                for e in self.archConvParamsDisc:
                    f.write(f"{int(e)},")
                f.write(f'\n')   
                f.write(f'Architecture Den - Units\n')
                for e in self.archDen:
                    f.write(f"{int(e)},")
                f.write('\n')
                f.write(f'Architecture Den - Activation function\n')
                for e in self.archDenParamsDisc:
                    f.write(f"{int(e)},")
                f.write('\n')
                f.write(f'Total de parametros\n')
                f.write(f'{totalParams}\n')
                f.write(f'Accuracy - Val\n')
                f.write(f'{score_val:.4f}\n')
                f.write(f'Accuracy - Test\n')
                f.write(f'{score_test:.4f}\n')
                         
    def savePlot(self, gen, sampleN, popType, score_val, hist):
        plt.plot(hist.epoch, hist.history['loss'], 'g', label='loss')
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{dirMaker}{gen}/model{sampleN}-{popType}:it_{gen},score_{score_val:.4f}.png')
        plt.clf()        

    def saveModel(self, gen, sampleN, popType, model, score_val, ):
        model.save(f'{dirMaker}{gen}/model{sampleN}-{popType}:it_{gen},score_{score_val:.4f}.h5')

class GA:
    def __init__(self, sizePopulation, sizeTournament, history=True, save=True, historySamples=10, debug=False, verbose=False):
        self.population = []
        self.newPopulation = []
        self.sizePopulation = sizePopulation
        self.sizeTournament = sizeTournament
        self.populationFit = []
        self.populationFitTest = []
        self.populationParams = []
        self.newPopulationFit = []
        self.newPopulationFitTest = []
        self.newPopulationParams = []
        self.gen=0
        self.history = history
        self.save = save
        self.historySamples= historySamples
        self.debug = debug
        self.verbose = verbose
        
        os.mkdir(dirMaker+f'0')

        for i in range(self.sizePopulation):
            self.population.append(sample())
            self.newPopulation.append(sample())

        for i in range(self.sizePopulation):
            if self.debug:
                self.populationFit.append(rn.uniform(0,1))
                self.populationFitTest.append(rn.uniform(0,1))
                self.populationParams.append(rn.randint(50000, 100000))

                self.newPopulationFit.append(rn.uniform(0,1))
                self.newPopulationFitTest.append(rn.uniform(0,1))
                self.newPopulationParams.append(rn.randint(50000, 100000))
            else:
                model, score_test, score_val, hist = self.population[i].fitnessModel(self.verbose)
                self.populationFit.append(score_val)
                self.populationFitTest.append(score_test)
                params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
                self.populationParams.append(params)
            
                model, score_test, score_val, hist = self.newPopulation[i].fitnessModel(self.verbose)
                self.newPopulationFit.append(score_val)
                self.newPopulationFitTest.append(score_test)
                params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
                self.newPopulationParams.append(params)

        fullPopulation, fullPopulationFit, fullPopulationFitTest, fullPopulationParams = self.sortPopulationsByFit()
        self.setPopulation(fullPopulation, fullPopulationFit, fullPopulationFitTest, fullPopulationParams)
        if self.save:
            self.saveGeneration()
        if self.history:
            self.makeHistory()
        print('###### GS Inicializado ######')

    def tournament(self):
        tournament = []
        tournamentFitness = []
        index = rn.sample((range(0, self.sizePopulation-1)), self.sizeTournament)
        for t in range(self.sizeTournament):
            tournament.append(self.population[index[t]])
            tournamentFitness.append(self.populationFit[index[t]])
        firstParent = np.argmax(tournamentFitness)
        tournamentFitness[firstParent] /= 2 
        secondParent = np.argmax(tournamentFitness)
        return tournament[firstParent], tournament[secondParent]

    def crossArithParams(self, fpParamsCont, spParamsCont, fpParamsDisc, spParamsDisc):
        global paramsCont, paramsDisc
        childParamsCont = np.zeros(shape=fpParamsCont.shape[0])
        for i in range(fpParamsCont.shape[0]):
            alpha = rn.uniform(0,1)
            aux = alpha * fpParamsCont[i] + (1-alpha) * spParamsCont[i]
            if aux >= paramsCont[i, 0] and aux <= paramsCont[i, 1]:
                childParamsCont[i] = aux
            else: 
                ran = rn.uniform(0,1)
                if ran > 0.5:
                    childParamsCont[i] = fpParamsCont[i]
                else: 
                    childParamsCont[i] = spParamsCont[i]

        childParamsDisc = np.zeros(shape=fpParamsDisc.shape[0])
        for i in range(fpParamsDisc.shape[0]):
            binaryFPR = format(int(fpParamsDisc[i]), "03b")
            binarySPR = format(int(spParamsDisc[i]), "03b")
            binaryChild = ""
            for j in range(len(binaryFPR)):
                ran = rn.uniform(0,1)
                if ran > 0.5:
                    binaryChild+=binaryFPR[j]
                else:
                    binaryChild+=binarySPR[j]
            childParamsDisc[i] = int(binaryChild, 2)       

        return childParamsCont, childParamsDisc

    def populationCross(self):
        for i in range(self.sizePopulation):
            firstParent, secondParent = self.tournament()
            fpParamsCont, fpParamsDisc = firstParent.getParams()
            spParamsCont, spParamsDisc = secondParent.getParams()
            fpArchConv, fpArchConvK, fpArchConvParams, fpArchDen, fpArchDenParams = firstParent.getArch()
            spArchConv, spArchConvK, spArchConvParams, spArchDen, spArchDenParams = secondParent.getArch()
            newParamsCont, newParamsDisc = self.crossArithParams(fpParamsCont, spParamsCont, fpParamsDisc, spParamsDisc)
            newArchConv, newArchConvK, newArchDen, newArchConvParamsDisc, newArchDenParamsDisc = self.crossArithArch(
                fpArchConvParams, spArchConvParams, fpArchDenParams, spArchDenParams, fpArchConv, spArchConv,
                fpArchConvK, spArchConvK, fpArchDen, spArchDen
            )
            self.newPopulation[i].setSample(newParamsCont, newParamsDisc, newArchConv, newArchConvK, newArchDen, newArchConvParamsDisc, newArchDenParamsDisc)

    def crossArithArch(self, fpParamsArchConv, spParamsArchConv, fpParamsArchDen, 
                       spParamsArchDen, fpArchConv, spArchConv, fpArchConvK, spArchConvK, fpArchDen, spArchDen):
        global paramsArchConvDisc, paramsArchDenDisc, archConv, archDen
        childConvParams = np.zeros(shape=(fpParamsArchConv.shape))
        childDenParams = np.zeros(shape=(fpParamsArchDen.shape))
        childConvArch = np.zeros(shape=(fpArchConv.shape))
        childConvArchK = np.zeros(shape=(fpArchConvK.shape))
        childDenArch = np.zeros(shape=(fpArchDen.shape))
        for i in range(fpParamsArchConv.shape[0]):
            binaryFPR = format(int(fpParamsArchConv[i]), "03b")
            binarySPR = format(int(spParamsArchConv[i]), "03b")
            binaryChild = ""
            for j in range(len(binaryFPR)):
                ran = rn.uniform(0,1)
                if ran > 0.5:
                    binaryChild+=binaryFPR[j]
                else:
                    binaryChild+=binarySPR[j]

            childConvParams[i] = int(binaryChild, 2)

        for i in range(fpParamsArchDen.shape[0]):
            binaryFPR = format(int(fpParamsArchDen[i]), "03b")
            binarySPR = format(int(spParamsArchDen[i]), "03b")
            binaryChild = ""
            
            for j in range(len(binaryFPR)):
                ran = rn.uniform(0,1)
                if ran > 0.5:
                    binaryChild+=binaryFPR[j]
                else:
                    binaryChild+=binarySPR[j]

            childDenParams[i] = int(binaryChild, 2)
        
        for i in range(fpArchConv.shape[0]):
            alpha = rn.uniform(0,1)
            aux = alpha * fpArchConv[i] + (1-alpha) * spArchConv[i]
            aux2 = alpha * fpArchConvK[i] + (1-alpha) * spArchConvK[i]
            if aux >= archConv[0] and aux <= archConv[1]:
                childConvArch[i] = aux
            else: 
                ran = rn.uniform(0,1)
                if ran > 0.5:
                    childConvArch[i] = fpArchConv[i]
                else: 
                    childConvArch[i] = spArchConv[i]

            if aux2 >= archConv[3] and aux2 <= archConv[4]:
                childConvArchK[i] = aux2
            else: 
                ran = rn.uniform(0,1)
                if ran > 0.5:
                    childConvArchK[i] = fpArchConvK[i]
                else: 
                    childConvArchK[i] = spArchConvK[i]
        
        for i in range(fpArchDen.shape[0]):
            alpha = rn.uniform(0,1)
            aux = alpha * fpArchDen[i] + (1-alpha) * spArchDen[i]
            if aux >= archDen[0] and aux <= archDen[1]:
                childDenArch[i] = aux
            else: 
                ran = rn.uniform(0,1)
                if ran > 0.5:
                    childDenArch[i] = fpArchDen[i]
                else: 
                    childDenArch[i] = spArchDen[i]

        return childConvArch, childConvArchK, childDenArch, childConvParams, childDenParams

    def mutateArch(self):
        global archConv, mutationProb
        for i in range(self.sizePopulation):
            mutation = rn.uniform(0,1)
            if mutation <= mutationProb:
                archConv, archConvK, archConvPD, archDen, archDenPD = self.newPopulation[i].getArch()
                newArchConv = np.random.permutation(archConv)
                newArchConvK = np.random.permutation(archConvK)
                newArchDen = np.random.permutation(archDen)
                self.newPopulation[i].setArch(newArchConv, newArchConvK, newArchDen)          

    def populationMutate(self):
        self.mutateArch()

    def sortPopulationsByFit(self): 
        fullPopulation = self.population + self.newPopulation
        fullPopulationFit = self.populationFit + self.newPopulationFit
        fullPopulationFitTest = self.populationFitTest + self.newPopulationFitTest
        fullPopulationParams = self.populationParams + self.newPopulationParams
        permutation = np.argsort(np.array(fullPopulationFit)) # minor to major
        newFullPopulation = []
        newFullPopulationFit = []
        newFullPopulationFitTest = []
        newFullPopulationParams = []
        for i in range(permutation.shape[0]):
            newFullPopulation.append(fullPopulation[permutation[i]])
            newFullPopulationFit.append(fullPopulationFit[permutation[i]])
            newFullPopulationFitTest.append(fullPopulationFitTest[permutation[i]])
            newFullPopulationParams.append(fullPopulationParams[permutation[i]])
        return newFullPopulation, newFullPopulationFit, newFullPopulationFitTest, newFullPopulationParams
    
    def setPopulation(self, fullPopulation, fullPopulationFit, fullPopulationFitTest, fullPopulationParams):
        self.population = fullPopulation[int(self.sizePopulation):]
        self.populationFit = fullPopulationFit[int(self.sizePopulation):]
        self.populationFitTest = fullPopulationFitTest[int(self.sizePopulation):]
        self.populationParams = fullPopulationParams[int(self.sizePopulation):]

    def evalPop(self):
        if self.debug:
            self.populationFit = []
            self.populationFitTest = []
            self.populationParams = []

            self.newPopulationFit = []
            self.newPopulationFitTest = []
            self.newPopulationParams = []

        for i in range(self.sizePopulation):
            if self.debug:
                self.populationFit.append(rn.uniform(0,1))
                self.populationFitTest.append(rn.uniform(0,1))
                self.populationParams.append(rn.randint(50000, 100000))

                self.newPopulationFit.append(rn.uniform(0,1))
                self.newPopulationFitTest.append(rn.uniform(0,1))
                self.newPopulationParams.append(rn.randint(50000, 100000))
            else:
                model, score_test, score_val, hist = self.newPopulation[i].fitnessModel(self.verbose)
                self.newPopulationFit[i] = score_val 
                self.newPopulationFitTest[i] = score_test   
                params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
                self.newPopulationParams[i] = params
    
    def saveGeneration(self):
        for i in range(self.sizePopulation):
            self.population[i].save(
                self.gen, 
                i, 
                f'population', 
                self.populationParams[i], 
                self.populationFitTest[i], 
                self.populationFit[i])
            self.newPopulation[i].save(
                self.gen,
                i, 
                f'offspring', 
                self.newPopulationParams[i],
                self.newPopulationFitTest[i],
                self.newPopulationFit[i])

    def makeHistory(self):
        with open(f'{dirMaker}{self.gen}/history.txt', 'w') as f:
                f.write(f'Top 10 Individuals.\n')
                for i in range(1, self.historySamples+1):
                    
                    sampleConvArch, _, _, sampleDenArch, _ = self.population[self.sizePopulation - i].getArch()

                    f.write(f'#####  ##### Sample:{i} #####  #####\n')
                    f.write(f'Sample Accuracy Validation: {self.populationFit[self.sizePopulation - i]:.4f}\n')
                    f.write(f'Sample Accuracy Test: {self.populationFitTest[self.sizePopulation - i]:.4f}\n')
                    f.write(f'Sample Parameters: {self.populationParams[self.sizePopulation - i]}\n')
                    f.write(f'Architecture Conv - Kernels: [')
                    for e in sampleConvArch:
                        f.write(f"{int(e)},")
                    f.write(f']\n')
                    f.write(f'Architecture Den - Units: [')
                    for e in sampleDenArch:
                        f.write(f'{int(e)},')
                    f.write(f']\n')
                    f.write(f'#####  ##### ###### #####  #####\n')

    def runGeneticAlgorithm(self, generations=10):
        print('###### Inicia Algoritmo ######')
        for gen in range(generations):
            self.gen+=1
            os.mkdir(dirMaker+f'{self.gen}')
            print(f'###### Inicia generación:{self.gen} ######')
            self.populationCross()
            print(f'###### Cruza exitosa:{self.gen} ######')
            self.mutateArch()
            print(f'###### Mutacion exitosa:{self.gen} ######')
            self.evalPop()
            fullPopulation, fullPopulationFit, fullPopulationFitTest, fullPopulationParams = self.sortPopulationsByFit()
            self.setPopulation(fullPopulation, fullPopulationFit, fullPopulationFitTest, fullPopulationParams)
            if self.save:
                self.saveGeneration()
                print(f'###### Generación salvada:{self.gen} ######')
            if self.history:
                self.makeHistory()
                print(f'###### Historial creado:{self.gen} ######')
            print(f'###### Evaluación Exitosa, gen:{self.gen} ######')
            
        return

    def showPopulations(self):
        for i in range(self.sizePopulation):
            print(f'#### Population: sample {i} ####')
            self.population[i].show()
            print(f'Acc: {self.populationFit[i]}')
        
        for i in range(self.sizePopulation):
            print(f'#### NewPopulation: sample {i} ####')
            self.newPopulation[i].show()
            print(f'Acc: {self.newPopulationFit[i]}')

    def showMatingPool(self):
        for i in range(self.matingPoolSize):
            print(f'#### MatingPool: sample {i} ####')
            self.matingPool[i].show()
            print(f'Acc: {self.matingPoolFit[i]}')

worker = GA(
    sizePopulation=40, 
    sizeTournament=5, 
    history=True,
    save=True,
    historySamples=10,
    debug=False,
    verbose=False)

worker.runGeneticAlgorithm(generations = 25)







'''
for i in range(10):
    GeneticAlgorithm.runGeneticAlgorithm(
        tol = 0.1,
        generations = 15,
        stopCondition=False,
        probCross=1,
        probMuta=1/25,
        showBestPerGeneration=True,
        showPopulationsPerGeneration=False,
        genHistory=True,
        name=f'{i}.txt')
'''