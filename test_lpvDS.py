from nonlinear_ds import *
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
import mousetrajectory_gui as mt

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

'''
 Brings up an empty world environment with the drawn trajectories using MouseTrajectory GUI
 and the learned LPV-DS model
'''

if __name__ == '__main__':

    # Create figure/environment to draw trajectories on
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)    
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('LPV-DS learned from drawn trajectories:',fontsize=15)
          
    # Load trajectories from file and plot
    models_dir = './models/'
    model_name = 'test1.yml'
    lpv_ds = lpv_DS(filename=models_dir+model_name,order_type='F')
    
    # Draw vector field from learned lpv-ds
    grid = 40
    for i in np.linspace(-0.25, 1.25, grid):
    	for j in np.linspace(0, 1, grid):
    		x          = np.array([i, j])
    		x_dot      = lpv_ds.get_ds(x)
    		x_dot_norm = x_dot/LA.norm(x_dot) * 0.02
    		plt.arrow(i, j, x_dot_norm[0], x_dot_norm[1], 
				head_width=0.008, head_length=0.01)    

    file_name = './data/human_demonstrated_trajectories.dat'
    l,t,x,y   = mt.load_trajectories(file_name)
    ax.plot(x, y, 'ro', markersize=2, lw=2)

    # Show
    plt.show()

    

