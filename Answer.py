import scipy as sc
import numpy as np
from matplotlib import cm
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# X = (hours sleeping, hours studying, hours exercising), y = Score on test

X = np.array(([3,5,1], [5,1,1], [10,2,1]), dtype=float)
y = np.array(([752], [822], [293]), dtype=float)

RowsForTesting = 3
a = np.matrix(np.genfromtxt(str('input.txt'), delimiter=',', dtype=float, comments='#', filling_values="0",skip_footer=RowsForTesting,usecols=(0,1,2)))
b = np.matrix(np.genfromtxt(str('input.txt'), delimiter=',', dtype=float, comments='#', filling_values="0",skip_footer=RowsForTesting,usecols=(3))).T

X[:]= a
y[:]=b

# Scaling
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100


class Neural_Network(object):
    def __init__(self,Lambda=0): 
        self.Lambda=Lambda       
        #Define Hyperparameters
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        np.random.seed(0)  
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.hiddenLayerSize)
        self.W3 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2) 
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3,delta4)
        
        delta3 = np.dot(delta4, self.W3.T)*self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X, delta2)
        
        return dJdW1, dJdW2, dJdW3
    
    def getParams(self):
        #Get W1 , W2 and W3 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1, W2 and W3 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize,self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.hiddenLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],(self.hiddenLayerSize,self.hiddenLayerSize))
        W3_end = W2_end+ self.hiddenLayerSize*self.outputLayerSize
        self.W3=np.reshape(params[W2_end:W3_end],(self.hiddenLayerSize,self.outputLayerSize))

        
    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))

    def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad

#To train Nueral Network - trainer class
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


#-------function calling------------------------
        
NN= Neural_Network(Lambda=0.0001)

cost1 = NN.costFunction(X,y)
dJdW1,dJdW2,dJdW3 = NN.costFunctionPrime(X,y)

scalar = 3
NN.W1 = NN.W1 + scalar * dJdW1
NN.W2 = NN.W2 + scalar * dJdW2
NN.W3 = NN.W3 + scalar * dJdW3
cost2 = NN.costFunction(X,y)
print(cost1, cost2)

dJdW1,dJdW2,dJdW3 = NN.costFunctionPrime(X,y)
scalar = 3
NN.W1 = NN.W1 - scalar * dJdW1
NN.W2 = NN.W2 - scalar * dJdW2
NN.W3 = NN.W3 - scalar * dJdW3
cost3 = NN.costFunction(X,y)
print(cost2, cost3)

#Assign cost1 value for mincost variable
mincost=cost1

#While loop for find minimum cost using Gradient Decendent 
while True:
    NN = Neural_Network()
    cost1 = NN.costFunction(X,y)
    dJdW1,dJdW2,dJdW3 = NN.costFunctionPrime(X,y)
    
    scalar = 3
    NN.W1 = NN.W1 + scalar * dJdW1
    NN.W2 = NN.W2 + scalar * dJdW2
    NN.W3 = NN.W3 + scalar * dJdW3
    cost2 = NN.costFunction(X,y)
    #print(cost1, cost2)
    dJdW1,dJdW2,dJdW3 = NN.costFunctionPrime(X,y)
    
    scalar = 3
    NN.W1 = NN.W1 - scalar * dJdW1
    NN.W2 = NN.W2 - scalar * dJdW2
    NN.W3 = NN.W3 - scalar * dJdW3
    cost3 = NN.costFunction(X,y)
    #print(cost2, cost3)
    
    if(cost3>=cost1) & (cost2>=cost1):
        if mincost<=cost1:
            break
    else:
        mincost=cost1
        

#Training dataset
        
T = trainer(NN)
T.train(X,y)
print ("After train predicted test score")
print (NN.forward(X))
print ("y")
print (y)

#Drawing 3D graph
hoursSleep=np.linspace(0,10,100)
hoursStudy=np.linspace(0,10,100)
hoursExercising=np.linspace(0,10,100)

hoursSleepNorm=hoursSleep/10.
hoursStudyNorm=hoursStudy/3.
hoursExercisingNorm=hoursExercising/5.

a,b,c=np.meshgrid(hoursSleepNorm,hoursStudyNorm,hoursExercisingNorm)

#merge
allInputs=np.zeros((a.size,3))
allInputs[:,0]=a.ravel()
allInputs[:,1]=b.ravel()
allInputs[:,2]=c.ravel()

allOutputs=NN.forward(allInputs)
yy=np.dot(hoursStudy.reshape(100,1),np.ones((1,100)))
xx=np.dot(hoursSleep.reshape(100,1),np.ones((1,100)))
#(1)
plt.plot(T.J)
plt.grid(1)
plt.title("Marks Analysis")
plt.ylabel("Cost")
plt.xlabel("Iterations")
#(2)
#fig=plt.figure()
#ax= fig.gca(projection='3d')
#ax.scatter(X[:,0],X[:,1],y,c='k',alpha=1,s=30)
#surf= ax.plot_surface(xx,yy,allOutputs.reshape(100,100),cmap=plt.cm.jet,alpha=0.5,rstride=10, cstride=10)
#ax.set_xlabel('Hours Slept')
#ax.set_ylabel('Hours Studied')
#ax.set_zlabel('Test Score')

#(3)
fig = plt.figure()
ax = fig.gca(projection='3d')               # to work in 3d
plt.hold(True)

x_surf=np.arange(0, 1, 0.01)                # generate a mesh
y_surf=np.arange(0, 1, 0.01)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = np.sqrt(x_surf+y_surf)           # ex. function, which depends on x and y
ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot);    # plot a 3d surface plot

n = 100
np.random.seed(0)                                     # seed let us to have a reproducible set of random numbers
x=[X[:,0]]              # generate n random points
y=[X[:,1]]
z=[X[:,2]]
ax.scatter(x, y, z);                        # plot a 3d scatter plot

ax.set_xlabel('Hours Slept')
ax.set_ylabel('Error(y)')
ax.set_zlabel('Test Score')

plt.show()

#==Testing data Set==
testX = np.array(([4,5,1], [4,1,1], [9,2,1]), dtype=float)
testy = np.array(([70], [89], [85]), dtype=float)

c = np.matrix(np.genfromtxt(str('input.txt'), delimiter=',', dtype=float, comments='#', filling_values="0",usecols=(0,1,2))[-RowsForTesting:,:])
d = np.matrix(np.genfromtxt(str('input.txt'), delimiter=',', dtype=float, comments='#', filling_values="0",usecols=(3))[-RowsForTesting:]).T

testX[:]= c
testy[:]=d

print ("Testing data")
print (NN.forward(testX))
print (testy)

