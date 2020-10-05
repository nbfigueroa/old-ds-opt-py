#!/usr/bin/env python
#!/usr/bin/env python3

import numpy as np
import numpy.linalg as LA

def ds_eulerIntegration(ds_fun, dt, x0_all, attractor, max_iter, sim_tol):
    '''
     Generates forward integrated trajectories with the given DS from using 
     first-order Euler integration method
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
