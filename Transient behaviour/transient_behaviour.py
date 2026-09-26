import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Constants ---
Pem = 5
Peh = 5
gam = 25
B = 0.5
beta = 2.5
yw = 1

n = 250
eta = np.linspace(0, 1, n)
dx = eta[1] - eta[0]

def R(y, Da):
    """
    This function calculates the residual for the steady state system.
    Returns the residual vector
    """
    R_vec = np.zeros(2*n)
    # Boundary conditions
    R_vec[0] = y[0] - 1.0
    R_vec[n] = y[n] - 1.0
    R_vec[n-1] = (y[n-2] - y[n-1])/dx
    R_vec[2*n-1] = (y[2*n-2] - y[2*n-1])/dx

    # Interior nodes
    for i in range(1, n-1):
        R_vec[i] = (1/Pem)*(y[i+1]-2*y[i]+y[i-1])/dx**2 - (y[i+1]-y[i-1])/(2*dx) \
                   - Da*y[i]*np.exp(gam - gam/y[n+i])
        R_vec[n+i] = (1/Peh)*(y[n+i+1]-2*y[n+i]+y[n+i-1])/dx**2 - (y[n+i+1]-y[n+i-1])/(2*dx) \
                     - beta*(y[n+i]-yw) + B*Da*y[i]*np.exp(gam - gam/y[n+i])
    return R_vec

def jacobian_R(y, Da):
    """
    This function calculates the Jacobian matrix for the steady state system.
    Retruns the Jacobian matrix
    """
    J = np.zeros((2*n, 2*n))
    # Boundary
    J[0,0] = 1.0
    J[n,n] = 1.0
    J[n-1,n-2] = 1/dx
    J[n-1,n-1] = -1/dx
    J[2*n-1,2*n-2] = 1/dx
    J[2*n-1,2*n-1] = -1/dx

    for i in range(1, n-1):
        J[i,i-1] = 1/(Pem*dx**2) + 1/(2*dx)
        J[i,i] = -2/(Pem*dx**2) - Da*np.exp(gam - gam/y[n+i])
        J[i,i+1] = 1/(Pem*dx**2) - 1/(2*dx)
        J[i,n+i] = -Da*y[i]*gam*(y[n+i]**-2)*np.exp(gam - gam/y[n+i])

        J[n+i,i] = B*Da*np.exp(gam - gam/y[n+i])
        J[n+i,n+i-1] = 1/(Peh*dx**2) + 1/(2*dx)
        J[n+i,n+i] = -2/(Peh*dx**2) - beta + B*Da*y[i]*gam*(y[n+i]**-2)*np.exp(gam - gam/y[n+i])
        J[n+i,n+i+1] = 1/(Peh*dx**2) - 1/(2*dx)
    return J

"""
WORKING ON THIS FUNCTION
"""
def transient_sys(tau, y, Da):

    """
    This function evaluates the transient system for a specified Da value.
    Returns the rhs vector of the pde system (dependent on time)
    """
    #separate both variables into their dynamic states
    #y_i,1 -->y_i,n-2
    y1_dyn = y[1:n-2] #slice from y1,1 up to y1,n-2, skip initial and boundary conditions
    y2_dyn = y[n-2:-2] #slice from y2,1 up to y2,2*(2-2) skip initial and boundary conditions

    #initialize rhs vector 
    dydtau = np.zeros(2*(n-2))

    #separate both diff eq results
    dy1dtau = dydtau[:n-2]
    dy2dtau = dydtau[n-2:]    

    for i in range(2, n-2): 
        dy1dtau[i] = (1/Pem)*(y1_dyn[i+1]-2*y1_dyn[i]+y1_dyn[i-1])/dx**2 - (y1_dyn[i+1]-y1_dyn[i-1])/(2*dx) \
                   - Da*y1_dyn[i]*np.exp(gam - gam/y1_dyn[n+i])
        dy2dtau[i] = (1/Peh)*(y2_dyn[n+i+1]-2*y2_dyn[n+i]+y2_dyn[n+i-1])/dx**2 - (y2_dyn[n+i+1]-y2_dyn[n+i-1])/(2*dx) \
                     - beta*(y2_dyn[n+i]-yw) + B*Da*y2_dyn[i]*np.exp(gam - gam/y2_dyn[n+i])
 
    return dydtau