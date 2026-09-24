import numpy as np
import matplotlib.pyplot as plt

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

# --- Residuals ---
def R(y, Da):
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

# --- Jacobian of R ---
def jacobian_R(y, Da):
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

# --- dR/dDa ---
def dR_dDa(y):
    R_Da = np.zeros(2*n)
    for i in range(1,n-1):
        R_Da[i] = -y[i]*np.exp(gam - gam/y[n+i])
        R_Da[n+i] = B*y[i]*np.exp(gam - gam/y[n+i])
    return R_Da

# --- Augmented system ---
def F_aug(y, Da, y_k, Da_k, dot_y, dot_Da, delta_s):
    F = np.zeros(2*n+1)
    F[:2*n] = R(y, Da)
    F[-1] = np.dot(y - y_k, dot_y) + (Da - Da_k)*dot_Da - delta_s
    return F

def jacobian_aug(y, Da, dot_y, dot_Da):
    JF = np.zeros((2*n+1, 2*n+1))
    J_y = jacobian_R(y, Da)
    R_Da = dR_dDa(y)
    JF[:2*n, :2*n] = J_y
    JF[:2*n, -1] = R_Da
    JF[-1, :2*n] = dot_y
    JF[-1, -1] = dot_Da
    return JF

# --- Tangent vector (robust using least-squares) ---
def compute_tangent(y, Da, prev_dot_y=None, prev_dot_Da=None):
    J_y = jacobian_R(y, Da)
    R_Da = dR_dDa(y)
    dot_Da = 1.0
    dot_y, _, _, _ = np.linalg.lstsq(J_y, -R_Da*dot_Da, rcond=None)
    # Normalize
    norm = np.sqrt(np.dot(dot_y,dot_y) + dot_Da**2)
    dot_y /= norm
    dot_Da /= norm
    # Maintain branch direction
    if prev_dot_y is not None and np.dot(dot_y, prev_dot_y) + dot_Da*prev_dot_Da < 0:
        dot_y = -dot_y
        dot_Da = -dot_Da
    return dot_y, dot_Da

# --- Continuation parameters ---
delta_s = 1e-3
Da_min = 0.1
Da_max = 0.3

y0 = np.ones(2*n) 
Da0 = Da_min 

y_k = y0.copy() 
Da_k = Da0 

branch_y = [] 
branch_Da = []

prev_dot_y = None 
prev_dot_Da = None 
max_iterations = 10000  # max iteration safeguard
iteration_count = 0

try:
    while Da_k <= Da_max:

        if Da_k > Da_max:
            print(f"Da_k exceeded maximum ({Da_k}). Stopping iteration.")
            break

        # --- Compute tangent ---
        dot_y, dot_Da = compute_tangent(y_k, Da_k, prev_dot_y, prev_dot_Da)
        prev_dot_y, prev_dot_Da = dot_y.copy(), dot_Da

        # --- Predictor ---
        y_pred = y_k + delta_s*dot_y
        Da_pred = Da_k + delta_s*dot_Da

        # --- Corrector (Newton-Raphson) ---
        y_corr = y_pred.copy()
        Da_corr = Da_pred
        for it in range(100):
            F = F_aug(y_corr, Da_corr, y_k, Da_k, dot_y, dot_Da, delta_s)
            if np.linalg.norm(F) < 1e-8:
                break
            JF = jacobian_aug(y_corr, Da_corr, dot_y, dot_Da)
            # Damped Newton
            delta = np.linalg.solve(JF, -F)
            lambda_factor = 0.3
            y_corr += lambda_factor*delta[:2*n]
            Da_corr += lambda_factor*delta[-1]

        branch_y.append(y_corr.copy())
        branch_Da.append(Da_corr)

        # --- Log current Da ---
        with open("continuation_backup.txt", "a") as f:
            f.write(f"{Da_corr}\n")

        # --- Update for next step ---
        y_k = y_corr
        Da_k = Da_corr
        print(f"Da_k = {Da_k:.6f}")

except KeyboardInterrupt:
    print("Interrupted by user. Saving results before exit...")
    np.save("saved_branch_y.npy", branch_y)
    np.save("saved_branch_Da.npy", branch_Da)
    print("Results saved.")

    # ---- SAFEGUARD / LOGGING ---- 
    with open("continuation_backup.txt", "a") as f:
        f.write(f"{Da_corr}\n")  #type: ignore

        # Update for next step 
        y_k = y_corr #type: ignore
        Da_k = Da_corr #type: ignore
        print(f"Da_k = {Da_k:.6f}") 

# --- Convert to arrays --- 
branch_y = np.array(branch_y) 
branch_Da = np.array(branch_Da) 

# --- Plot y1 and y2 along branch --- 
plt.figure() 
for i in range(0, len(branch_y), max(1,len(branch_y)//20)): 
    plt.plot(eta, branch_y[i,:n], label=f'Da={branch_Da[i]:.3f}') 
plt.xlabel('eta') 
plt.ylabel('y1') 
plt.title('y1 along solution branch') 
plt.legend() 
plt.show()

plt.figure() 
for i in range(0, len(branch_y), max(1,len(branch_y)//20)): 
    plt.plot(eta, branch_y[i,n:], label=f'Da={branch_Da[i]:.3f}') 
plt.xlabel('eta') 
plt.ylabel('y2') 
plt.title('y2 along solution branch')
plt.legend() 
plt.show() 

# --- Bifurcation diagram at eta=1 --- 
plt.figure() 
plt.plot(branch_Da, branch_y[:, n-1], label='y1(eta=1)')
plt.plot(branch_Da, branch_y[:, 2*n-1], label='y2(eta=1)')
plt.xlabel('Da') 
plt.ylabel('y at eta=1') 
plt.title('Bifurcation diagram at outlet (eta=1)') 
plt.legend() 
plt.show() 

plt.plot(branch_Da)
plt.xlabel("Continuation step")
plt.ylabel("Da")
plt.show()