import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# --- Constants ---
Pem = 5
Peh = 5
gam = 25
B = 0.5
beta = 2.5
yw = 1
tau_final = 20

n = 250
eta = np.linspace(0, 1, n)
deta = eta[1] - eta[0]

def transient_sys(tau, y, Da):

    """
    This function evaluates the transient system for a specified Da value.
    Returns the rhs vector of the pde system (dependent on time)
    """
    #extract dynamic states (variables that change with time)
    y1_dyn = y[:n-2]
    y2_dyn = y[n-2:]

    #reconstruct spatial vectors 
    y1 = np.zeros(n)
    y2 = np.zeros(n)

    #define kinterior dynamic states
    y1[1:n-1] = y1_dyn
    y2[1:n-1] = y2_dyn

    #define inlet boundary conditions
    y1[0] = (
        (Pem*2*deta + 4*y1[1] - y1[2])/
        (3 + Pem*2*deta)
    )
    y2[0] = (
        (Peh*2*deta + 4*y2[1] - y2[2])/
        (3 + Peh*2*deta)
    )

    #define outlet boundary conditions
    y1[-1] = (4/3)*y1[-2] - (1/3)*y1[-3]
    y2[-1] = (4/3)*y2[-2] - (1/3)*y2[-3]

    #define rhs vector
    dydtau = np.zeros(2*(n-2))

    #split rhs vector for easier handling
    dy1dtau = dydtau[:n-2]
    dy2dtau = dydtau[n-2:]

    #discretize pde in interior nodes
    for i in range(1, n-1):

        dy1dtau[i-1] = (
            (1/Pem)
            * (y1[i+1] - 2*y1[i] + y1[i-1])
            / deta**2
            - (y1[i+1] - y1[i-1])
            / (2*deta)
            - Da*y1[i]*np.exp(gam - gam/y2[i])
        )

        dy2dtau[i-1] = (
            (1/Peh)
            * (y2[i+1] - 2*y2[i] + y2[i-1])
            / deta**2
            - (y2[i+1] - y2[i-1])
            / (2*deta)
            - beta*(y2[i] - yw)
            + B*Da*y1[i]*np.exp(gam - gam/y2[i])
        )

    return dydtau

#uniform initial concentration in the reactor
y1_dyn_initial = np.ones(n-2)
y2_dyn_initial = np.ones(n-2)

y0 = np.concatenate([
    y1_dyn_initial,
    y2_dyn_initial
])

def system(check_err=False):
    """
    This function calls the solve_ivp function and produces the time-dependant
    y1 and y2 profiles.
    
    If check_err is set to true, solve_ivp is called with varying tolerances in order 
    to calculate the numerical convergence of the system 
    """


    #define evaluation time 
    t_eval = np.linspace(0, tau_final, 1000)


    if check_err == True:
        #initialize Da_lst 
        Da_lst= np.linspace(0.16,0.18,5) #less Da values to save time

        #solution list 
        sol_lst_err_comparison= []

        #tolerance lists
        rtol_lst = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
        atol_lst = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

        for tol in range(len(rtol_lst)):
            rtol = rtol_lst[tol]
            atol = atol_lst[tol]

            for Da in Da_lst:
                print(f"Current Da = {Da}")
                sol = solve_ivp(
                    fun=transient_sys,
                    t_span=(0, tau_final),
                    y0=y0,
                    t_eval=t_eval,
                    args=(Da,),
                    method="BDF", #problem is stiff,
                    rtol = rtol,
                    atol = atol
                )   

                sol_lst_err_comparison.append(sol)
                print(f"Solutions for Da = {Da} found!")

        

    else:
        #initialize Da_lst and solution lst
        Da_lst= np.linspace(0.16,0.18,10) #more Da values 
        transient_solutions = []

        for Da in Da_lst:
            print(f"Current Da = {Da}")
            sol = solve_ivp(
                fun=transient_sys,
                t_span=(0, tau_final),
                y0=y0,
                t_eval=t_eval,
                args=(Da,),
                method="BDF" #problem is stiff
            )

            transient_solutions.append(sol)
            print(f"Solutions for Da = {Da} found!")

        #reaction axial length is controlled by eta parameter [0,1], and is discretized by 250
        #nodes
        i = n // 2 #change integer division to see results at different points in the reactor.

        #initialize plots
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for j in range(len(Da_lst)):
        
            sol = transient_solutions[j]

            #slicing here comes from the offset between yi_dyn and 
            #physical node (which includes IC at the start and BC at the end)
            y1_transient = sol.y[i-1, :] 
            y2_transient = sol.y[n-2+ (i-1),:]

            Da = Da_lst[j] #for labelling 

            #plotting for y1
            axes[0].plot(
                sol.t,
                y1_transient,
                label=rf"$Da={Da:.3f}$"
            )

            axes[0].set_title(rf"$y_1(\eta,\tau)$ at $\eta={eta[i]:.2f}$")
            axes[0].set_ylabel(r"$y_1$")
            axes[0].set_xlabel(r"$\tau$")

            #plotting for y2
            axes[1].plot(
                sol.t,
                y2_transient,
                label=rf"$Da={Da:.3f}$"
            )

            axes[1].set_title(rf"$y_2(\eta,\tau)$ at $\eta={eta[i]:.2f}$")
            axes[1].set_ylabel(r"$y_2$")
            axes[1].set_xlabel(r"$\tau$")
            
        #save plots using path
        file_path = (
            Path.cwd()
            / 'Transient behaviour'
            / 'Transient behaviour.png'
        )

        fig.legend(
        bbox_to_anchor=(0.5, 0.02),
        loc='lower center',
        ncol=5
)

        fig.tight_layout(rect=(0, 0.25, 1, 1))

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches='tight',
            transparent=False
        )

system(check_err=False)


    


