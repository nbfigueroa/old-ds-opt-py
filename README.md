# ds-opt-py
Python version of the [ds-opt](https://github.com/nbfigueroa/ds-opt) package to generate, learn and execute dynamical system motion generators with the LPV-DS formulation.


Currently this is only an interface between Python and MATLAB, hence, [ds-opt](https://github.com/nbfigueroa/ds-opt) (and all of its dependencies) must be in your MATLAB path. Next steps are to implement the GMM-sampler and SDP optimization in python.


Dependencies: 
```
numpy, scipy, matplotlib
```
Some of the plotting functions use latex code for labels. If you don't have latex installed in your OS you should remove these lines. 
