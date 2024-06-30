import numpy as np

class PCAfunc (object):
    def __init__(self,data1,k):
        self.D = data1
        self.K = k
        self.centreD = []
        self.x=[]
        self.tz=[]
        self.u=[]
        self.z=[]

        self.centreD = self.centerlized()
        self.x = self.cov()
        self.u = self.U()
        self.z = []


    def centerlized(self):
        mean = [np.mean(i) for i in self.D.T]

        centreD = self.D - mean
        return centreD

    def cov(self):
        TOTAL=np.shape(self.D)[0]
        x=np.dot(self.D,self.D.T)/(TOTAL-1)
        return x
    def U(self):
        tz,tx=np.linalg.eig(self.x)


        P=np.argsort(-1*tz)
        UT=[tx[:,P[i]] for i in range(k)]
        U=np.transpose(UT)
        print( self.K, U)
        return U
    def Z(self):
        Z=self.centreD*self.u
        print(Z)
        return Z










if __name__ =="__main__":
    data1 = np.array([[1,3,5],
                      [10,20,5],
                      [8,5,15],
                      [16,4,13],
                      [8,3,1]])
    k = np.shape(data1)[1] - 1
    pca=PCAfunc(data1,k)