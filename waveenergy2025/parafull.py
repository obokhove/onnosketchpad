import firedrake as fd
import numpy as np

# Define Li, gam, GZZ12, a_rad, Zbar, alp, Hm
# Parameters coil
a_rad = 0.012 # % coil outer radius was 0.04 now 24mm diameter
mu0 = 4*np.pi*10**(-7)    # % permeability of vacuum
L = 0.076 # total coil
L3 = L/3 # one of three coils
Kf = 1/(1+0.528*(2*a_rad/L)**(0.846))
Kf3 = 1/(1+0.528*(2*a_rad/L3)**(0.846))
Kf = 0.880 # from Nagaoka's table
Kf3 = 0.701 # from Nagaoka's table
Kf = 0.880
N = 2499
N3 = 2499/3 # 833 per coil
Li = Kf*np.pi*a_rad**2*mu0*N**2/L # coil induction
Li3 = Kf3*np.pi*a_rad**2*mu0*N3**2/L3 # coil induction Li3 = 0.006
Ri = 0  # 
Rc = 3*15  #
alp = 1  # 
Hm = 0.04  # 
nq = 1.0  # 
Vt = 2.05  #
Isat = 0.02  #
Rl = nq*Vt/Isat
Rl = 100 # 100
Rc3 = Rc/3
Ri3 = Ri/3
Rl3 = Rl/3
m = 2.988*3  # 25-02-2025 3 magnets
m = 3*2.33516 # reduces to 11kg on steel
muu = mu0*m/(4*np.pi)     
gam = 2*np.pi*a_rad**2*muu*N/L
gam3 = 2*np.pi*a_rad**2*muu*N3/L3
