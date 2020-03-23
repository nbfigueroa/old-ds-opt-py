#!/usr/bin/env python2

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

            self.n_gaussians = int(yaml_dict['K'])
            self.n_dimensions = int(yaml_dict['M'])

            self.Mu = np.reshape(yaml_dict['Mu'], (self.n_dimensions, self.n_gaussians), order=order_type)
            self.Priors = np.array(yaml_dict['Priors'])
            self.Sigma = np.reshape(yaml_dict['Sigma'], (self.n_dimensions, self.n_dimensions, self.n_gaussians), order=order_type)

            self.A_g = np.reshape(yaml_dict['A'], (self.n_dimensions, self.n_dimensions, self.n_gaussians), order=order_type)

            if hasattr(yaml_dict, 'b'):
                self.b_g = np.reshape(yaml_dict['b'], (self.n_dimensions, self.n_gaussians), order=order_type)
            else:
                self.b_g = np.zeros((self.n_dimensions, self.n_gaussians))    

            self.starting_point_teaching = np.reshape(yaml_dict['x0_all'], (self.n_dimensions, -1), order=order_type)
            self.mean_starting_point = np.mean(self.starting_point_teaching, axis=1)

            self.attractor = yaml_dict['attractor']

            # print("TODO -- import lvpDS from file")
            # import pdb; pdb.set_trace() # DEBUG
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
        

    def posterior_probs_gmm(self, x, normalization_type='norm'):
        n_datapoints = np.array(x).shape[1]

        # Compute mixing weights for multiple dynamics
        Px_k = np.zeros((self.n_gaussians, n_datapoints))

        # Compute probabilities p(x^i|k)
        for k in range(self.n_gaussians):
            
            Px_k[k,:] = self.ml_gaussPDF(x, self.Mu[:,k], self.Sigma[:,:,k]) + REALMIN

        ### Compute posterior probabilities p(k|x) -- FAST WAY --- ###
        alpha_Px_k = np.tile(self.Priors.T, (1, n_datapoints))*Px_k

        if normalization_type=='norm':
            Pk_x = alpha_Px_k / np.tile(np.sum(alpha_Px_k, axis=0), (self.n_gaussians, 1))
        elif normalization_type=='un-norm':
            Pk_x = alpha_Px_k

        return Pk_x


    def ml_gaussPDF(self, Data, Mu, Sigma):
        #ML_GAUSSPDF
        # This def computes the Probability Density Def (PDF) of a
        # multivariate Gaussian represented by means and covariance matrix.
        #
        # Author:	Sylvain Calinon, 2009
        #			http://programming-by-demonstration.org
        #
        # Inputs -----------------------------------------------------------------
        #   o Data:  D x N array representing N datapoints of D dimensions.
        #   o Mu:    D x 1 array representing the centers of the K GMM components.
        #   o Sigma: D x D x 1 array representing the covariance matrices of the 
        #            K GMM components.
        # Outputs ----------------------------------------------------------------
        #   o prob:  1 x N array representing the probabilities for the 
        #            N datapoints.     
        # (nbVar,notate) = np.array(Data).shape

        #      (D x N) - repmat((D x 1),1,N)
        #      (D x N) - (D x N)
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



# if __name__=="__main__":
#     print("Example simulation")

#     import_dir = "./models/"
#     import_file = "record_ft_a_v3_ds0.yml"
#     lvp_ds = []
#     lvp_ds.append(lpv_DS(filename=import_dir+import_file))
#     x = np.array([[1],
#                   [3]])
#     lvp_ds[0].get_ds(x)
