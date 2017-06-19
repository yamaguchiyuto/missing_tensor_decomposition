import math
import itertools

import numpy as np

class MODEL:
    def __init__(self,R,lamb,data_shape,nepochs=30,verbose=0):
        self.R = R
        self.lamb = lamb
        self.data_shape = data_shape
        self.M = len(data_shape) # number of indices
        self.nepochs = nepochs
        self.verbose=verbose

    def _init_q(self):
        q = {}
        for n in range(self._X.shape[0]):
            for m in range(self.M):
                if self._missing(n,m):
                    if not n in q: q[n] = {}
                    q[n][m] = np.random.random(self.data_shape[m])
                    q[n][m] /= np.sum(q[n][m])
        return q

    def _missing(self,n,m):
        if self._X[n,m]<-100:
            return True
        else:
            return False

    def _init_U(self):
        U = {}
        mean = abs(self._X[:,-1].sum() / (self.R*self._N))**(1.0/len(self.data_shape))
        for m in range(self.M):
            U[m] = np.random.normal(loc=0, scale=np.sqrt(1/self.lamb), size=(self.data_shape[m], self.R))
        return U

    def _init_EU(self):
        self._EU = {}
        for n in self.q:
            for m in self.q[n]:
                self._EU[(n,m)] = self._calc_EU(n,m)
        return self._EU

    def _init_EUU(self):
        self._EUU = {}
        for n in self.q:
            for m in self.q[n]:
                self._EUU[(n,m)] = self._calc_EUU(n,m)
        return self._EUU

    def get_EU(self,n,m):
        if self._missing(n,m):
            return self._EU[(n,m)]
        else:
            return self.A[m][self._get_index(n,m)]

    def _get_index(self,n,m):
        return int(self._X[n,m])

    def get_EUU(self,n,m):
        if self._missing(n,m):
            return self._EUU[(n,m)]
        else:
            return np.outer(self.A[m][self._get_index(n,m)],self.A[m][self._get_index(n,m)])

    def _calc_EU(self,n,m):
        EU = self.q[n][m].dot(self.A[m])
        return EU

    def _calc_EUU(self,n,m):
        EUU = (self.A[m].T*self.q[n][m]).dot(self.A[m])
        return EUU

    def _update_EU(self,n,m):
        self._EU[(n,m)] = self._calc_EU(n,m)

    def _update_EUU(self,n,m):
        self._EUU[(n,m)] = self._calc_EUU(n,m)

    def _do_estep(self):
        for n in self.q:
            self.remained_estep_iter = 1
            while True:
                if self._estep_converge(): break
                for m in self.q[n]:
                    a = self._calc_a(n,m)
                    a -= np.max(a)
                    expa_p = np.exp(a)
                    z = np.sum(expa_p)
                    self.q[n][m] = expa_p/z
                    self._update_EU(n,m)
                    self._update_EUU(n,m)

    def _calc_a(self,n,m):
        EUEU = np.prod([self.get_EU(n,_m) for _m in range(self.M) if _m!=m], axis=0) #R-dim
        EUUEUU = np.prod([self.get_EUU(n,_m) for _m in range(self.M) if _m!=m], axis=0) #R^2-dim
        UEUUEUUU = np.sum(self.A[m].dot(EUUEUU)*self.A[m],axis=1)
        a = -0.5*UEUUEUUU
        if self._get_value(n) != 0:
            a += self._get_value(n) * self.A[m].dot(EUEU)
        return a

    def _get_value(self,n):
        return self._X[n,-1]

    def _get_index(self,n,m):
        return int(self._X[n,m])

    def _do_mstep(self):
        modes = np.arange(self.M)
        while True:
            if self._mstep_converge(): break
            np.random.shuffle(modes)
            for m in modes:
                G = self._calc_lssol(m)
                self.A[m] = G
                for n in self.q:
                    if self._missing(n,m):
                        self._update_EU(n,m)
                        self._update_EUU(n,m)

    def _calc_lssol(self,m):
        G = np.zeros((self.data_shape[m],self.R,self.R))
        H = np.zeros_like(self.A[m])
        for n in range(self._N):
            value = self._get_value(n)
            z = np.prod([self.get_EU(n,_m) for _m in range(self.M) if _m!=m], axis=0)
            zz = np.prod([self.get_EUU(n,_m) for _m in range(self.M) if _m!=m], axis=0)
            if self._missing(n,m):
                H += value * np.outer(self.q[n][m],z)
                G += np.outer(self.q[n][m],zz).reshape((self.data_shape[m],self.R,self.R))
            else:
                i = int(self._X[n,m])
                H[i] += value*z
                G[i] += zz

        Sol = np.zeros_like(self.A[m])
        for i in range(self.data_shape[m]):
            Sol[i] = np.linalg.solve((G[i]+self.lamb*np.identity(self.R)), H[i])
        return Sol

    def _converge(self):
        if self.remained_iter<=0:
            return True
        else:
            self.remained_iter-=1
            return False

    def _estep_converge(self):
        if self.remained_estep_iter<=0:
            return True
        else:
            self.remained_estep_iter-=1
            return False
    def _mstep_converge(self):
        if self.remained_mstep_iter<=0:
            return True
        else:
            self.remained_mstep_iter-=1
            return False

    def _discard_incomplete_samples(self,X):
        indices = np.where(np.sum(X[:,:-1]<0,axis=1)==0)[0] # indices of complete samples
        return X[indices]

    def _discard_incomplete_modes(self,X):
        complete_modes = np.where(np.sum(X[:,:-1]<0,axis=0)==0)[0] # complete modes
        self.data_shape = self.data_shape[complete_modes]
        self.M = len(self.data_shape)
        complete_modes = np.hstack([complete_modes, -1])
        return X[:,complete_modes]

    def fit(self,X):
        self._X = X
        self._N = self._X.shape[0] # number of samples
        self.q = self._init_q()
        self.A = self._init_U()
        self._init_EU()
        self._init_EUU()
        self.remained_iter = self.nepochs
        while True:
            if self.verbose:
                print(self.remained_iter)
            if self._converge(): break
            self._do_estep()
            self.remained_mstep_iter = 1
            self._do_mstep()
        return self

    def predict(self,indices): # CP model
        return np.sum(np.prod([self.A[m][int(indices[m])] for m in range(self.M)],axis=0))
