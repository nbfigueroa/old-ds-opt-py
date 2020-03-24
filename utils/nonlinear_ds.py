import numpy as np
import numpy.linalg as LA
from numpy import pi
import time, yaml

REALMIN = np.finfo(np.double).tiny


class DynamicalSystem():
    def __init__(self):
        print("Define empty base-constructor")

    def get_ds(self):
        return [0,0]

    def is_attractor_reached(self):
        return None


class lpv_DS(DynamicalSystem):
    def __init__(self,  debug = 0, A_g=[], b_g=[], Mu=[], Priors=[], Sigma=[], filename="", order_type='F'):
        # order_type for yaml list2matrix-encoding. Use "F" for MATLAB created file.
        t0 = time.time()    
        self.debug = debug
        if len(filename):
            yaml_dict = yaml.load(open(filename))

            # LPV-DS variables
            self.K         = int(yaml_dict['K'])                    
            self.M         = int(yaml_dict['M'])            
            self.Mu        = np.reshape(yaml_dict['Mu'], (self.M, self.K), order=order_type)
            self.Priors    = np.array(yaml_dict['Priors'])
            self.Sigma     = np.reshape(yaml_dict['Sigma'], (self.M, self.M, self.K), order=order_type)
            self.A_g       = np.reshape(yaml_dict['A'], (self.M, self.M, self.K), order=order_type)
            self.attractor = np.array(yaml_dict['attractor'])            
            
            # Auxiliary variables
            if hasattr(yaml_dict, 'b'):
                self.b_g = np.reshape(yaml_dict['b'], (self.M, self.K), order=order_type)
            else:
                self.b_g = np.zeros((self.M, self.K))    
                for k in range(self.K):
                    self.b_g[:,k] = -self.A_g[:,:,k].dot(self.attractor)
            self.x0_all = np.reshape(yaml_dict['x0_all'], (self.M, -1), order=order_type)
            self.dt  = float(yaml_dict['dt'])

            if self.debug:
                # For debugging purposes
                print('K=',self.K)
                print('M=',self.M)
                print('Mu=',self.Mu)
                print('Priors=',self.Priors)
                print('Sigma=',self.Sigma)
                print('A=',self.Sigma)
                print('b=',self.b_g)
                print('attractor=', self.attractor)
                print('x0_all=',self.x0_all)
                print('dt=',self.dt)
        
        else:

            # LPV-DS variables
            self.M = np.array(x).shape[0] 
            self.K = len(self.Priors)
            self.A_g       = A_g
            self.b_g       = b_g
            self.Mu        = Mu
            self.Priors    = Priors
            self.Sigma     = Sigma
            self.attractor = np.zeros((dim))

        # Posterior Probabilities per local DS
        tF = time.time()
        print("Time to initialize {} s". format(tF-t0))


    @property 
    def dt(self):
        return self.__dt

    @property 
    def x0_all(self):
        return self.__x0_all

    @property 
    def attractor(self):
        return self.__attractor

    @attractor.setter 
    def attractor(self, attractor):
        self.__attractor = attractor

    def is_attractor_reached(self, x, margin_attr=0.1):
        return np.sqrt((x-self.attractor)**2) < margin_attr 
    

    
    def get_ds(self, x, normalization_type='norm'):
        x = np.array(x)
        if len(np.array(x).shape) ==1:
            x = x.reshape(2,1)
        
        n_datapoints = x.shape[1]

        beta_k_x = self.posterior_probs_gmm(x, normalization_type)

        # Output Velocity
        x_dot = np.zeros((self.M, n_datapoints))
        for i in range(np.array(x).shape[1]):
            # Estimate Global Dynamics component as LPV
            if np.array(self.b_g).shape[1] > 1:
                f_g = np.zeros((self.M, self.K))
                for k in range(self.K):                    
                    # For a fixed DS
                    f_g[:,k] = beta_k_x[k,i] * (self.A_g[:,:,k].dot(x[:,i]) + self.b_g[:,k])

                    # For a transformable DS
                    # f_g[:,k] = beta_k_x[k,i] * (self.A_g[:,:,k].dot(x[:,i] - self.attractor))

                f_g = np.sum(f_g, axis=1)
            else:
                # Estimate Global Dynamics component as Linear DS
                f_g = (self.A_g*x[:,i] + self.b_g)

            x_dot[:,i] = f_g

        # Allow transformation of DS
        # self.transform_ds(x_dot)           
        return x_dot.squeeze()
    
    def posterior_probs_gmm(self, x, normalization_type='norm'):
        n_datapoints = np.array(x).shape[1]

        # Compute mixing weights for multiple dynamics
        Px_k = np.zeros((self.K, n_datapoints))

        # Compute probabilities p(x^i|k)
        for k in range(self.K):
            
            Px_k[k,:] = self.gaussPDF(x, self.Mu[:,k], self.Sigma[:,:,k]) + REALMIN

        ### Compute posterior probabilities p(k|x) -- FAST WAY --- ###
        alpha_Px_k = np.tile(self.Priors.T, (1, n_datapoints))*Px_k

        if normalization_type=='norm':
            Pk_x = alpha_Px_k / np.tile(np.sum(alpha_Px_k, axis=0), (self.K, 1))
        elif normalization_type=='un-norm':
            Pk_x = alpha_Px_k

        return Pk_x


    def gaussPDF(self, Data, Mu, Sigma):
        n_datapoints = np.array(Data).shape[1]
        if n_datapoints>1:
            warnings.warn("WARNING --- Check tile")

        Mus  = np.tile(Mu, (n_datapoints, 1)).T
        Data = (Data - Mus)
        prob = np.sum((Data.T.dot(LA.inv(Sigma))).dot(Data), axis=1)            
        prob = np.exp(-0.5*prob) / np.sqrt((2*pi)**self.M * (np.abs(LA.det(Sigma))+REALMIN)) + REALMIN

        return prob

    def set_transform_ds(self, rotation=None, strecthing=None, translation=None):
        if rotation==None:
            self.rotation = np.diag((self.M))
        else:
            self.rotation = rotation
            
        if strecthing:
            self.strecthing = np.diag((self.M))
        else:
            self.strecthing = strecthing
            
        if translation==None:
            self.translation = np.diag((self.M))
        else:
            self.translation = translation        
        
        
    def transform_ds(self, x_dot):
        # x_dot as reference!
        x_dot = self.rotation*(self.stretching*(self.translation+x_dot))
