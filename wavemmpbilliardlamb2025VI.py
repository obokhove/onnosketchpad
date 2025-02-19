import firedrake as fd
from firedrake import (
    min_value
)
from petsc4py import PETSc  # Import PETSc to access NINFINITY and INFINITY
import math
from math import *
import time as tijd
import numpy as np
import sympy as sp
from sympy import summation
from sympy.abc import k
import matplotlib.pyplot as plt
import os
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule

# 
Lx = 2.0 #
Ly = 1.5
nx = 1   # 
nCG = 1     # function space order horizontal
nFmaxfunc = 0

# control parameters
save_path =  "Ineqcbilliard" 
if not os.path.exists(save_path):
    os.makedirs(save_path)                                       

#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
tsize2 = 12
size = 16  # font size of image axes
t = 0

#________________________ MESH  _______________________#
# 
mesh = fd.IntervalMesh(nx, Lx)
x = fd.SpatialCoordinate(mesh)
x = mesh.coordinates

t0 = 0.0
nic = 1
time = []
t = 0
if nic == 1: 
    Tperiod = 40
    nTfac = 1
    t_end = nTfac*Tperiod # time of simulation [s]
    Tstartmeas = 0.0
    dtt = np.minimum(0.001,0.05) # i.e. 
    Nt = 1 
    CFL = 0.5
    dt = CFL*dtt # CFL*dtt
    # print('dtt=, t_end/dtt, t_end',dtt, t_end/dtt, t_end)
        
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 200 # np.min(100,int(t_end/dtt))
    X00 = 0.0
    Y00 = 0.0
    U00 = 0.2
    V00 = 1.0
    Xm0 = 0.0
    Ym0 = 0.0
    lambm120 = 0
    bb = 0.0
    gamm = 100
    bb = 0.34*np.sqrt(gamm)
    nn = 3
    twon = 2*nn # print('bb**2/gamm',bb**2/gamm)
    
dtmeet = t_end/nplot # 
tmeet = dtmeet
tmeas = np.linspace(0.0, t_end, nplot+1)
epsmeet = 10.0**(-10)
nt = int(len(time)/nplot)
t_plot = time[0::nt]
# print(' dtmeet, tmeet', dtmeet, tmeet) # print('tmeas', tmeas) # print('nt',nt,len(time))

##_________________  FIGURE SETTINGS __________________________##
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# First subplot (ax1) - Scatter plot
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]
# 
ax1.set_title(r'Free particle billiard in squircle: MMP Firedrake',fontsize=tsize2)   
ax1.set_xlabel(r'$X(t)$ ',fontsize=size)
ax1.set_ylabel(r'$Y(t)$ ',fontsize=size)
ax2.set_xlabel(r'$t$ ',fontsize=size)
ax2.set_ylabel(r'$(H(t)-H(0))/H(0)$ ',fontsize=size)
ax3.set_ylabel(r'$U(t)$ ',fontsize=size)
ax3.set_xlabel(r'$t$ ',fontsize=size)
ax4.set_xlabel(r'$t$ ',fontsize=size)
ax4.set_ylabel(r'$G(X,Y)$ ',fontsize=size)

#__________________  Define function spaces  __________________#
nDG = 0
V_W = fd.FunctionSpace(mesh, 'DG', nDG, vfamily='DG') # needs "aij" in solver parameters
V_C = fd.FunctionSpace(mesh, 'R', 0, vfamily='R', vdegree=0) # billiard variables X, U, Y and V; needs "nest" in solver parameters

# Variables for modified midpoint test case billiard
mixed_Vmpc = V_C * V_C * V_C * V_C * V_C
result_mixedmpc = fd.Function(mixed_Vmpc)
vvmpc = fd.TestFunction(mixed_Vmpc)
vvmpc0, vvmpc1, vvmpc2, vvmpc3, vvmpc4 = fd.split(vvmpc) # These represent "blocks".
Xh12, Uh12, Yh12, Vh12,lamb12 = fd.split(result_mixedmpc)
lam12 = fd.Function(V_C)
X1 = fd.Function(V_C)
U1 = fd.Function(V_C)
Y1 = fd.Function(V_C)
V1 = fd.Function(V_C)
X0 = fd.Function(V_C)
U0 = fd.Function(V_C)
Y0 = fd.Function(V_C)
V0 = fd.Function(V_C)

##_________________  Initial Conditions __________________________##
X0.assign(X00)
U0.assign(U00)
Y0.assign(Y00)
V0.assign(V00) #print('t, X00, U00 Y00 V00',t,X00,U00,Y00,V00)
X0p = X00 # np.array(X0.vector())
U0p = U00 # np.array(U0.vector())
Y0p = Y00 # np.array(Y0.vector())
V0p = V00 # np.array(V0.vector())
ax3.plot(t,U0p,'.k')
ax3.plot(t,V0p,'.r')
t = 0.0
i = 0.0
E0 = 0.5*(U0p**2+V0p**2)

# MMP https://epubs.siam.org/doi/10.1137/1.9781611976311
vpolyp = 5

solver_parameters9 = {
    "snes_type": "vinewtonrsls",  # Projected Newton method for variational inequality
    "mat_type": "nest",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_0_fields": "0,1,2,3",
    "pc_fieldsplit_1_fields": "4",
}
solver_parameters29 = {
    # "snes_type": "newtonls",  # Standard Newton solver for the whole system
    "mat_type": "aij",
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-12,
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_0_fields": "0,2",  # Fields solved normally
    "pc_fieldsplit_1_fields": "1,3,4",  # Fields 1,3,4 get the VI solver
    "pc_fieldsplit_1_snes_type": "vinewtonrsls"  # Apply VI solver only to split 1,3,4
}

VPnl = (1/Lx)*( Uh12*(X1-X0)-Xh12*(U1-U0)+Vh12*(Y1-Y0)-Yh12*(V1-V0)-0.5*dt*(Uh12**2 + Vh12**2) )*fd.dx(degree=vpolyp)  
X_expr = fd.derivative(VPnl, Uh12, du=vvmpc1) # du=v_C eqn for X1
U_expr = fd.derivative(VPnl, Xh12, du=vvmpc0)   # du=v_C eqn for U1
Y_expr = fd.derivative(VPnl, Vh12, du=vvmpc3) # du=v_C eqn for Y1
V_expr = fd.derivative(VPnl, Yh12, du=vvmpc2)  # du=v_C eqn for V1
X_expr = fd.replace(X_expr, {X1: 2*Xh12-X0}) # X1 = 2*Xh12-X0 
U_expr = fd.replace(U_expr, {U1: 2*Uh12-U0}) #
Y_expr = fd.replace(Y_expr, {Y1: 2*Yh12-Y0}) # Y1 = 2*Yh12-Y0 
V_expr = fd.replace(V_expr, {V1: 2*Vh12-V0}) #
G = 1 - ((2*Xh12 - X0)/Lx)**twon - ((2*Yh12 - Y0)/Ly)**twon
lamb_expr = (1/Lx)*(vvmpc4*( lamb12-G ) )*fd.dx(degree=vpolyp) # lamb=G>=0
Fexpr = X_expr+U_expr+Y_expr+V_expr+lamb_expr # +Gexpr
lbound = fd.Function(mixed_Vmpc).assign(PETSc.NINFINITY)
ubound = fd.Function(mixed_Vmpc).assign(PETSc.INFINITY)
ubound.sub(4).assign(PETSc.INFINITY)
lbound.sub(4).assign(0.0)
solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc), solver_parameters=solver_parameters9)

###### OUTPUT #########
# outfile_Z = fd.File("resultbuoy/billZ.pvd")
# Set-up exact outline squircles
NXX = 1000
XXp = np.linspace(-Lx,Lx,NXX)
YYpu = Ly*(1-(XXp/Lx)**twon)**(1/twon)
YYpm = -Ly*(1-(XXp/Lx)**twon)**(1/twon)

print('Time Loop starts')
tic = tijd.time()
while t <= 1.0*(t_end + dt): #
    solvelamb_nl.solve()
    Xh1, Uh12, Yh1, Vh12, lamb12 = fd.split(result_mixedmpc)
    X1.interpolate(2.0*Xh12-X0)
    Y1.interpolate(2.0*Yh12-Y0)
    U1.interpolate(2.0*Uh12-U0)
    V1.interpolate(2.0*Vh12-V0)
    lam12.interpolate(1.0*lamb12)
    X1p = np.array(X1.vector())
    X0p = np.array(X0.vector())
    U0p = np.array(U1.vector())
    Y0p = np.array(Y0.vector())
    Y1p = np.array(Y1.vector())
    V0p = np.array(V1.vector())
    lam0p = np.array(lam12.vector())
    t+= dt
    tmeet = tmeet+dtmeet
    if (t in t_plot): #
        ax1.plot(X0p,Y0p,'.k')
        ax3.plot(t,U0p,'.k')
        ax3.plot(t,V0p,'.r')
        E1 = 0.5*(U0p**2+V0p**2)  #
        print('Plotting starts, E1: ', E1, (E1-E0)/E0)        
        ax2.plot(t, (E1-E0)/E0, '.k')
        ax2_right = ax2.twinx()
        ax2_right.plot(t, E1, '.b', label="E1") #
        ax2_right.set_ylabel(r"H(t)", color='b')
        ax1.plot(XXp,YYpu,'-b')
        ax1.plot(XXp,YYpm,'-b')
        ax4.plot(t,lam0p,'.r')    
        plt.tight_layout()
        plt.pause(0.1)

    X0.assign(X1)
    U0.assign(U1)
    Y0.assign(Y1)
    V0.assign(V1)
# End while time loop      

toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
# print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
