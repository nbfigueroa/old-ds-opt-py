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
    # Simplify for single data point
    def __init__(self,  A_g=[], b_g=[], Mu=[], Priors=[], Sigma=[],filename="", order_type='F'):
        start_time = time.time()
        # order_type for yaml list2matrix-encoding. Use "F" for MATLAB created file.

        if len(filename):
            yaml_dict = yaml.load(open(filename))

            # Parse lpv-ds variables
            self.n_gaussians = int(yaml_dict['K'])
            print('K=',self.n_gaussians)
            
            self.n_dimensions = int(yaml_dict['M'])
            print('M=',self.n_dimensions)

            self.Mu = np.reshape(yaml_dict['Mu'], (self.n_dimensions, self.n_gaussians), order=order_type)
            print('Mu=',self.Mu)
            
            self.Priors = np.array(yaml_dict['Priors'])
            print('Priors=',self.Priors)
            
            self.Sigma = np.reshape(yaml_dict['Sigma'], (self.n_dimensions, self.n_dimensions, self.n_gaussians), order=order_type)
            print('Sigma=',self.Sigma)
            
            self.A_g = np.reshape(yaml_dict['A'], (self.n_dimensions, self.n_dimensions, self.n_gaussians), order=order_type)
            print('A=',self.Sigma)

            self.attractor = np.array(yaml_dict['attractor'])
            print('attractor=', self.attractor)

            self.starting_point_teaching = np.reshape(yaml_dict['x0_all'], (self.n_dimensions, -1), order=order_type)
            print('x0_all=',self.starting_point_teaching)


            # Auxiliary variables
            if hasattr(yaml_dict, 'b'):
                self.b_g = np.reshape(yaml_dict['b'], (self.n_dimensions, self.n_gaussians), order=order_type)
            else:
                self.b_g = np.zeros((self.n_dimensions, self.n_gaussians))    
                for k in range(self.n_gaussians):
                    self.b_g[:,k] = -self.A_g[:,:,k].dot(self.attractor)
            
            print('b=',self.b_g)
            self.mean_starting_point = np.mean(self.starting_point_teaching, axis=1)
            

        else:
            # TODO - check values
            self.A_g = A_g
            self.b_g = b_g

            self.Mu     = Mu
            self.Priors = Priors
            self.Sigma  = Sigma

            # Auxiliary Variables
            self.n_dimensions = np.array(x).shape[0] # 
            self.n_gaussians = len(self.Priors)

            self.mean_starting_point = np.ones((dim))
            self.attractor = np.zeros((dim))

            # Posterior Probabilities per local DS
        end_time = time.time()

        print("Time to initialize {} s". format(end_time-start_time))


    def is_attractor_reached(self, x, margin_attr=0.1):
        return np.sqrt((x-self.attractor)**2) < margin_attr 

    
    def get_ds(self, x, normalization_type='norm'):
        # [N,M] = size(x);
        x = np.array(x)
        if len(np.array(x).shape) ==1:
            x = x.reshape(2,1)
        
        n_datapoints = x.shape[1]

        beta_k_x = self.posterior_probs_gmm(x, normalization_type)

        # Output Velocity
        x_dot = np.zeros((self.n_dimensions, n_datapoints))
        for i in range(np.array(x).shape[1]):
            # Estimate Global Dynamics component as LPV
            if np.array(self.b_g).shape[1] > 1:
                f_g = np.zeros((self.n_dimensions, self.n_gaussians))
                for k in range(self.n_gaussians):
                    # import pdb; pdb.set_trace() ## DEBUG ##
                    
                    f_g[:,k] = beta_k_x[k,i] * (self.A_g[:,:,k].dot(x[:,i]) + self.b_g[:,k])

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
        Px_k = np.zeros((self.n_gaussians, n_datapoints))

        # Compute probabilities p(x^i|k)
        for k in range(self.n_gaussians):
            
            Px_k[k,:] = self.gaussPDF(x, self.Mu[:,k], self.Sigma[:,:,k]) + REALMIN

        ### Compute posterior probabilities p(k|x) -- FAST WAY --- ###
        alpha_Px_k = np.tile(self.Priors.T, (1, n_datapoints))*Px_k

        if normalization_type=='norm':
            Pk_x = alpha_Px_k / np.tile(np.sum(alpha_Px_k, axis=0), (self.n_gaussians, 1))
        elif normalization_type=='un-norm':
            Pk_x = alpha_Px_k

        return Pk_x


    def gaussPDF(self, Data, Mu, Sigma):
        n_datapoints = np.array(Data).shape[1]
        if n_datapoints>1:
            warnings.warn("WARNING --- Check tile")

        Mus  = np.tile(Mu, (n_datapoints, 1)).T
        Data = (Data - Mus)
        # Data = (N x D)
        # (N x 1)
        prob = np.sum((Data.T.dot(LA.inv(Sigma))).dot(Data), axis=1)            
        prob = np.exp(-0.5*prob) / np.sqrt((2*pi)**self.n_dimensions * (np.abs(LA.det(Sigma))+REALMIN)) + REALMIN

        return prob

    def set_transform_ds(self, rotation=None, strecthing=None, translation=None):
        if rotation==None:
            self.rotation = np.diag((self.n_dimensions))
        else:
            self.rotation = rotation
            
        if strecthing:
            self.strecthing = np.diag((self.n_dimensions))
        else:
            self.strecthing = strecthing
            
        if translation==None:
            self.translation = np.diag((self.n_dimensions))
        else:
            self.translation = translation        
        
        
    def transform_ds(self, x_dot):
        # x_dot as reference!
        x_dot = self.rotation*(self.stretching*(self.translation+x_dot))
