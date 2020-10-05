import sys
sys.path.append("./ds_tools/")
from nonlinear_ds import *
from modulation import *
from utils import *
import mousetrajectory_gui as mt

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

'''
 Brings up obstacle world environment and the modulated linear DS controller trajectories
'''

def demo_HBS():
    '''
    demo of the HBS approach with multiple obstacles
    '''
    x_target = np.array([0.9, 0.8])
    gamma1, gamma_grad1, obs_center1 = gamma_circle_2d(0.15, [0.2, 0.8])
    # gamma2, gamma_grad2, obs_center2 = gamma_circle_2d(0.15, [0.8, 0.2])
    gamma2, gamma_grad2, obs_center2 = gamma_cross_2d(0.1, 0.15, [0.7, 0.3])
    gamma3, gamma_grad3, obs_center3 = gamma_rectangle_2d(0.2, 0.3, [0.6, 0.7])
    gammas = [gamma1, gamma2, gamma3]
    gamma_grads = [gamma_grad1, gamma_grad2, gamma_grad3]
    obs_centers = [obs_center1, obs_center2, obs_center3]
    plt.figure()
    for i in np.linspace(0, 1, 30):
        for j in np.linspace(0, 1, 30):
            x = np.array([i, j])
            if min([g(x - oc) for g, oc in zip(gammas, obs_centers)]) < 1:
                continue

            x_dot = linear_controller(x, x_target)
            modulated_x_dot = modulation_HBS(x, x_dot, obs_centers, obs_centers, 
                gammas, gamma_grads) * 0.15

            plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1], 
                head_width=0.008, head_length=0.01)

    ##################################################################
    #   Integrate DS to generate forward trajectoriy from set of x0  #
    ##################################################################
    ds_fun     = lambda x: lpv_ds.get_ds(x)         
    dt         = lpv_ds.dt
    max_iter   = 5000
    sim_tol    = 0.005 
    x_sim      = ds_eulerIntegration(ds_fun, dt, x0_all, attractor, max_iter, sim_tol)


    ###############
    #   Plot all  #
    ###############
    plt.gca().add_artist(plt.Circle((0.2, 0.8), 0.15))
    # plt.gca().add_artist(plt.Circle((0.8, 0.2), 0.15))
    plt.gca().add_artist(plt.Rectangle((0.50, 0.25), 0.4, 0.1))
    plt.gca().add_artist(plt.Rectangle((0.65, 0.1), 0.1, 0.4))
    plt.gca().add_artist(plt.Rectangle((0.5, 0.55), 0.2, 0.3))
    plt.axis([0, 1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([x_target[0]], [x_target[1]], 'r*')
    plt.savefig('vector_field_HBS.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    x0_all = [0.2, 0.2]
    demo_HBS()
