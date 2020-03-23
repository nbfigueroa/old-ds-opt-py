from nonlinear_ds import *
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
import mousetrajectory_gui as mt
import matplotlib.gridspec as gridspec

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

'''
 Brings up obstacle world environment with the drawn trajectories using MouseTrajectory GUI
 and the learned LPV-DS model
'''

if __name__ == '__main__':


    ####################################################
    #   Load LPV-DS and trajectories used to learn it  #
    ####################################################
    models_dir = '../models/'
    model_name = 'test1-obst.yml'
    lpv_ds = lpv_DS(filename=models_dir+model_name,order_type='F')
    file_name = './data/human_demonstrated_trajectories_Mar22_23:06:56.dat'
    l,t,x,y   = mt.load_trajectories(file_name)

    ######################################################################
    #   Draw vector field from learned lpv-ds with quiver-like function  #
    ######################################################################
    # Create figure/environment to draw trajectories on
    fig, ax = plt.subplots()
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('LPV-DS learned from drawn trajectories:',fontsize=15)
    
    # Add Obstacles and target in environment
    x_target = np.array([0.9, 0.8])
    ax.plot(x_target[0], x_target[1], 'rd', markersize=10, lw=2)
    ax.add_artist(plt.Circle((0.2, 0.8), 0.15))
    ax.add_artist(plt.Rectangle((0.50, 0.25), 0.4, 0.1))
    ax.add_artist(plt.Rectangle((0.65, 0.1), 0.1, 0.4))
    ax.add_artist(plt.Rectangle((0.5, 0.55), 0.2, 0.3))
    
    # Add vector field from learned LPV-DS
    grid_size = 40
    for i in np.linspace(-0.25, 1.25, grid_size):
        for j in np.linspace(0, 1, grid_size):
            x_query    = np.array([i, j])
            x_dot      = lpv_ds.get_ds(x_query)
            x_dot_norm = x_dot/LA.norm(x_dot) * 0.02
            plt.arrow(i, j, x_dot_norm[0], x_dot_norm[1], 
                head_width=0.008, head_length=0.01)    

    # Add trajectories used to learn DS       
    ax.plot(x, y, 'ro', markersize=2, lw=2)


    ######################################################################
    #   Draw vector field from learned lpv-ds with streamline function   #
    ######################################################################
    
    # Create figure/environment to draw trajectories on
    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('LPV-DS learned from drawn trajectories:',fontsize=15)

    # Add Obstacles and target in environment
    x_target = np.array([0.9, 0.8])
    ax1.plot(x_target[0], x_target[1], 'rd', markersize=10, lw=2)
    ax1.add_artist(plt.Circle((0.2, 0.8), 0.15))
    ax1.add_artist(plt.Rectangle((0.50, 0.25), 0.4, 0.1))
    ax1.add_artist(plt.Rectangle((0.65, 0.1), 0.1, 0.4))
    ax1.add_artist(plt.Rectangle((0.5, 0.55), 0.2, 0.3))

    # Add streamline plot from learned LPV-DS
    grid_size = 50
    Y, X = np.mgrid[0:1:50j, -0.25:1.25:50j]
    V, U = np.mgrid[0:1:50j, -0.25:1.25:50j]
    for i in range(grid_size):
        for j in range(grid_size):
            x_query    = np.array([X[i,j], Y[i,j]])        
            x_dot      = lpv_ds.get_ds(x_query)
            x_dot_norm = x_dot/LA.norm(x_dot) * 0.02
            U[i,j]     = x_dot_norm[0]
            V[i,j]     = x_dot_norm[1]
  
    strm = ax1.streamplot(X, Y, U, V, density = 3.5, linewidth=0.55, color='k')

    # Add trajectories used to learn DS
    ax1.plot(x, y, 'ro', markersize=2, lw=2)


    # Show
    plt.show()


