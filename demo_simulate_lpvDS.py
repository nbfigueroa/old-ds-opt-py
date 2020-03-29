import sys
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Dynamical System Tools
from ds_tools.nonlinear_ds import *
import ds_tools.mousetrajectory_gui as mt
from scipy.integrate import odeint

# ODE integration from pytorch (will only work with python-3 and having torch installed)
include_torchdiffeq = 0
if sys.version_info.major == 3:
    import importlib.util
    package_name = 'torchdiffeq'
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(package_name +" is not installed")
    else:     
        from torchdiffeq import odeint
        include_torchdiffeq = 1

'''
 Brings up an empty world environment with the drawn trajectories using MouseTrajectory GUI
 and the learned LPV-DS model
'''

def ds_eulerIntegration(ds_fun, dt, x0_all, attractor, max_iter, sim_tol):
    '''
     Generates forward integrated trajectories with the given DS from using 
     first-order Euler integration method from multiple init conditions
    '''    
    dim, N_x0    = x0_all.shape        
    x_sim        = np.empty(shape=(N_x0*dim, max_iter))
    x_dot_sim    = np.empty(shape=(N_x0*dim, max_iter))
    sim_status = np.array([0]*N_x0)

    # Initial state
    for ii in range(N_x0):
        id0 = ii*dim 
        id1 = (ii+1)*dim 
        x_sim[id0:id1, 0] = np.array([x0_all[0,ii], x0_all[1,ii]])
    
    sim_iter   = 1
    while sim_status.sum() < N_x0:
        for ii in range(N_x0):
            id0 = ii*dim 
            id1 = (ii+1)*dim 
            x_dot_sim[id0:id1, sim_iter-1]  = ds_fun(x_sim[id0:id1, sim_iter-1])
            x_sim[id0:id1, sim_iter]        = x_sim[id0:id1, sim_iter-1] + x_dot_sim[id0:id1, sim_iter-1]*dt        
            if (sim_status[ii] == 0) & (LA.norm(x_sim[id0:id1, sim_iter]-attractor) < sim_tol):
                sim_status[ii] = 1
                print(ii, "-th simulation reached attractor")

        if sim_iter > max_iter-1:
            print(ii, "-th simulation reached maximum iterations")
            break

        sim_iter = sim_iter + 1
        
    return x_sim, x_dot_sim


if __name__ == '__main__':

    ####################################################
    #   Load LPV-DS and trajectories used to learn it  #
    ####################################################
    models_dir = './models/'

    #### Smooth snake test ####
    model_name = 'test1.yml'
    file_name = './data/human_demonstrated_trajectories.dat'
    
    #### Semi spiral shape ####
    model_name = 'test2.yml'
    file_name = './data/human_demonstrated_trajectories_Mar22_22:33:43.dat'

    # Load learned DS and trajectories
    lpv_ds     = lpv_DS(filename=models_dir+model_name,order_type='F')    
    l,t,x,y    = mt.load_trajectories(file_name)
    x0_all     = lpv_ds.get_x0all()
    attractor  = lpv_ds.get_attractor()

    # Options for simulation
    show_vector_field = True
    show_stream_lines = True
    simulate_ds_integ = True


    if simulate_ds_integ:
        ##################################################################
        #   Integrate DS to generate forward trajectoriy from set of x0  #
        ##################################################################
        ''' Solve a system of ordinary differential equations using Euler integration '''
        ds_fun     = lambda x: lpv_ds.get_ds(x)         
        dt         = lpv_ds.get_dt()
        max_iter   = 5000
        sim_tol    = 0.005 
        x_sim_eul  = ds_eulerIntegration(ds_fun, dt, x0_all, attractor, max_iter, sim_tol)

        '''Solve a system of ordinary differential equations using LSODA method
           LSODA: Adams/BDF method with automatic stiffness detection and switching [7], [8]. 
           This is a wrapper of the Fortran solver from ODEPACK. 
           To manually choose other methods we can use scipy.integrate.solve_ivp(..) instead
           For better accuracy we can also provide the Jacobian of the DS. 
        '''
        ds_fun_sci = lambda x,t: lpv_ds.get_ds(x)         
        int_time  = 5 #seconds 
        max_steps = round(int_time/dt)
        t         = np.linspace(0,int_time,max_steps)
        dim, N_x0 = x0_all.shape 
        x_sim_sci = np.empty(shape=(N_x0, max_steps, dim))
        for ii in range(N_x0):
            x_sim_sci[ii] = odeint(ds_fun_sci, x0_all[:,ii], t)
        
        '''  Solve a system of ordinary differential equations with the odeint() function 
             from the torchdiffeq code (https://github.com/rtqichen/torchdiffeq)
             Their approach sssumes ODE is non-stiff, not sure if that assumption 
             is valid for our type of DS as the ratio of eigenvalues of the Jacobian might be high,
             this can be checked!
             Solver types listed here: https://github.com/rtqichen/torchdiffeq/blob/master/README.md
        '''
        if sys.version_info.major == 3:
            class LambdaDS(nn.Module):
                def forward(self, t, x):
                    x_dot_np = ds_fun(y.numpy().transpose())
                    x_dot    = torch.tensor([[y_dot_np[0], y_dot_np[1]]])    
                    return x_dot
            t_torch     = torch.linspace(0., int_time, max_steps)
            x_sim_torch = odeint(LambdaDS(), torch.tensor([[x0_all[0,0], x0_all[1,0]]]), t_torch, method='dopri5')

    if show_stream_lines:
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

        # Add streamline plot from learned LPV-DS
        grid_size = 70
        Y, X = np.mgrid[0:1:70j, -0.25:1.25:70j]
        V, U = np.mgrid[0:1:70j, -0.25:1.25:70j]
        for i in range(grid_size):
            for j in range(grid_size):
                x_query    = np.array([X[i,j], Y[i,j]])        
                x_dot      = lpv_ds.get_ds(x_query)
                x_dot_norm = x_dot/LA.norm(x_dot) * 0.02
                U[i,j]     = x_dot_norm[0]
                V[i,j]     = x_dot_norm[1]
      
        strm = ax1.streamplot(X, Y, U, V, density = 3.5, linewidth=0.55, color='k')

        # Add trajectories used to learn DS
        ax1.plot(x, y, 'ro', markersize=1.5, lw=2)
        ax1.plot(attractor[0], attractor[1], 'md', markersize=12, lw=2)
        ax1.plot(x0_all[0,:], x0_all[1,:], 'gs', markersize=10, lw=2)

        # Add simulated trajectories used to learn DS            
        if simulate_ds_integ:            
            idx = 0
            for ii in range(N_x0):
                ax1.plot(x_sim_eul[0][idx,:], x_sim_eul[0][idx+1,:], 'bo', markersize=2, lw=2)
                idx=idx+dim

    if show_vector_field:
        ######################################################################
        #   Draw vector field from learned lpv-ds with quiver-like function  #
        ######################################################################
        # Create figure/environment to draw trajectories on
        fig0, ax0 = plt.subplots()
        ax0.set_xlim(-0.25, 1.25)
        ax0.set_ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('$x_1$',fontsize=15)
        plt.ylabel('$x_2$',fontsize=15)
        plt.title('LPV-DS learned from drawn trajectories:',fontsize=15)
           
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
        ax0.plot(x, y, 'ro', markersize=2, lw=2)
        ax0.plot(attractor[0], attractor[1], 'md', markersize=12, lw=2)
        ax0.plot(x0_all[0,:], x0_all[1,:], 'gs', markersize=10, lw=2)

        # Add simulated trajectories used to learn DS            
        if simulate_ds_integ:
            dim, N_x0  = x0_all.shape 
            idx = 0
            for ii in range(N_x0):
                ax0.plot(x_sim_eul[0][idx,:], x_sim_eul[0][idx+1,:], 'bo', markersize=2, lw=2)
                idx=idx+dim            
    # Show
    plt.show()
