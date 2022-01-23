from utils.solvers.vector import inner_prod
from utils.visualiser import visualise, visualise_neg, visualise_pos
from utils.solvers.solvers import getHyperPlaneFromTwoPoints
from utils.accuracy import accuracy as accuracy_
import numpy as np
from random import seed
np.random.seed(1)
seed(1)
class Node:
    def __init__(self, indim, solver, C = 1.0, rank = 3, xa = None, xb = None, constrain = 'lax', wnorm = 'L1'):
        self.weight = np.zeros(indim)
        self.bias = 0
        self.A = None
        self.B = None
        self.wA = 0
        self.wB = 0
        self.dim = indim
        self.C1=[]
        self.C2=[]
        self.C3=[]
        self.C4=[]
        self.labels = []
        self.X = []
        self.height = 0
        self.solver = solver
        self.C = C
        self.rank = rank
        self.xa = xa
        self.xb = xb
        self.constrain = constrain
        self.wnorm = wnorm
        
        
    def insert(self,neuron_type, weight=0, bias=0, w=0):     
        if neuron_type == 'A':
            self.A = Node(self.dim, self.solver, self.C, self.rank, self.xa, self.xb, self.constrain, self.wnorm)
            self.A.weight = weight
            self.A.bias = bias
            self.A.height = self.height+1
            return self.A
        else:
            self.B = Node(self.dim, self.solver, self.C, self.rank, self.xa, self.xb, self.constrain, self.wnorm)
            self.B.weight = weight
            self.B.bias = bias
            self.B.height = self.height+1
        return self.B
    
    def update_weights_and_bias(self,weight, bias, wA = 0, wB = 0):  
        self.weight = weight
        self.bias = bias
        self.wA = wA
        self.wB = wB
        
    def update_classes(self,ypred,ytrue):
        ypred=ypred.copy()
        ypred=np.reshape(ypred,(ypred.shape[0],1))    
        yf  = np.add(2*ypred, ytrue)
        self.C1 = np.argwhere(yf>2)[:,0] #1,1              #In order: predicted, true
        self.C2 = np.argwhere(yf<-2)[:,0] #-1,-1
        self.C3 = np.where((yf<2) & (yf>0))[0]   #1,-1
        self.C4 = np.where((yf<0) & (yf>-2))[0] #-1,1
        
    def forward(self, X): 
        
        y=[]
        X=X.copy()
        w = self.weight 
        b = self.bias
        wA = np.asarray([self.wA]).copy()
        wB=np.asarray([self.wB]).copy()

        if(self==None):
            return [] 
        if(self.A==None and self.B==None):
            y = np.sign(np.array(inner_prod(w, X))+np.array(b)).reshape(-1,1)
        if(self.A==None):
            xA = np.zeros((X.shape[0],1))
        else:
            xA = self.A.forward(X)
            xA=np.reshape(xA,(xA.shape[0],1))
        if(self.B==None):
            xB = np.zeros((X.shape[0],1)) 
        else:
            xB = self.B.forward(X)
            xB=np.reshape(xB,(xB.shape[0],1))
        if(self.A!=None and self.B!=None):
            wA = np.asarray([wA.item()])
            wB = np.asarray([wB.item()])
            y = np.sign(np.asarray(inner_prod(w, X))+np.asarray(inner_prod(wA, xA))+np.asarray(inner_prod(wB, xB))+np.asarray(b)).reshape(-1,1)
        if(self.A!=None and self.B==None):
            wA = np.asarray([wA.item()])
            y = np.sign(np.asarray(inner_prod(w, X))+np.asarray(inner_prod(wA, xA))+np.asarray(b)).reshape(-1,1)
        if(self.A==None and self.B!=None):
            wB = np.asarray([wB.item()])
            y = np.sign(np.asarray(inner_prod(w, X))+np.asarray(inner_prod(wB, xB))+np.asarray(b)).reshape(-1,1)
        return y
    
    def accuracy(self, xtrain, ytrain):
        return accuracy_(self.forward(xtrain),ytrain)
    
    
    def fine_tune_weights(self):
        l=self.labels.copy()
        X = self.X.copy()
        xA = np.zeros((X.shape[0],1))
        xB = np.zeros((X.shape[0],1))
        if(self==None):
            return   
        if(self.A!=None):
            self.A.fine_tune_weights()  
            xA = self.A.forward(X)
            xA=np.reshape(xA,(xA.shape[0],1))
        if(self.B!=None):
            self.B.fine_tune_weights() 
            xB = self.B.forward(X)
            xB=np.reshape(xB,(xB.shape[0],1))
        
        weight, bias, wA, wB = self.solver(X, l, self.C, self.rank, self.xa, self.xb, self.constrain, self.wnorm)
        
        print(self.height)
        visualise(weight,X.shape[1:])
        visualise_pos(weight,X.shape[1:])
        visualise_neg(weight,X.shape[1:]) 
        
        self.update_weights_and_bias(weight, bias, wA, wB)


    def recursive(self, X, labels, h):
        self.X = X
        self.labels = labels
        labels=labels.copy()
        X=X.copy()
        weight, bias, _, _1_ = self.solver(X, labels, self.C, self.rank, self.xa, self.xb, self.constrain, self.wnorm)
        self.update_weights_and_bias(weight, bias)
        ypred=self.forward(X)
        self.update_classes(ypred,labels)
        C1=self.C1
        C2=self.C2
        C3=self.C3
        C4=self.C4
        if(len(C3)==0 and len(C4)==0):
            return
        if(self.height>h-1):
            return
        if(len(C1)==0 or len(C2)==0):
            if(len(C1)!=0):
                X_positive=np.take(X,np.hstack((C1,C4)),axis=0) 
                X_negative=np.take(X,np.hstack((C3)),axis=0)
                #x1=X[C1[0]]
                #x2=X[C3[0]]
            elif(len(C2)!=0):
                X_positive=np.take(X,np.hstack((C4)),axis=0)  
                X_negative=np.take(X,np.hstack((C2,C3)),axis=0)
                #x1=X[C2[0]]
                #x2=X[C4[0]]
            else:
                #x1=X[C3[0]]
                #x2=X[C4[0]]
                X_positive=np.take(X,np.hstack((C4)),axis=0)  
                X_negative=np.take(X,np.hstack((C3)),axis=0)
            weight, bias = getHyperPlaneFromTwoPoints(X_positive, X_negative)
            self.update_weights_and_bias(weight, bias)
            ypred = self.forward(X)
            self.update_classes(ypred,labels)
            C1=self.C1
            C2=self.C2
            C3=self.C3
            C4=self.C4

        if(len(C3)!=0):
            X_new=np.take(X,np.hstack((C1,C3,C4)),axis=0)
            labels[C1]=-1
            labels[C3]=1
            labels[C4]=-1
            y_new=np.take(labels,np.hstack((C1,C3,C4)),axis=0)
            NodeA = self.insert('A')
            NodeA.recursive(X_new, y_new, h)
        if(len(C4) != 0):
            X_new=np.take(X,np.hstack((C2,C3,C4)),axis=0)
            labels[C2]=-1
            labels[C3]=-1
            labels[C4]=1
            y_new=np.take(labels,np.hstack((C2,C3,C4)),axis=0)
            NodeB = self.insert('B')
            NodeB.recursive(X_new, y_new, h)
