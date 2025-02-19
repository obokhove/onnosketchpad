import firedrake as fd
from firedrake import (
    min_value
)
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
# import os.path
# Run-time instructions: . /Users/amtob/firedrake/bin/activate or . /Users/onnobokhove/amtob/werk/firedrake/bin/activate
# parameters in SI units
# t_end  # time of simulation [s]
# dt = # time step [s]

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
    Tperiod = 20
    nTfac = 1
    t_end = nTfac*Tperiod # time of simulation [s]
    Tstartmeas = 0.0
    dtt = np.minimum(0.001,0.05) # i.e. 
    Nt = 1 # 
    CFL = 0.5
    dt = CFL*dtt # CFL*dtt
    print('dtt=, t_end/dtt, t_end',dtt, t_end/dtt, t_end)
        
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 200 # np.min(100,int(t_end/dtt))
    X00 = 0.0
    Y00 = 0.0
    U00 = 0.2
    V00 = 1.0
    bb = 0.0
    gamm = 100
    bb = 0.34*np.sqrt(gamm)
    nn = 3
    twon = 2*nn # 2*nn
    print('bb**2/gamm',bb**2/gamm)

    
dtmeet = t_end/nplot # 
tmeet = dtmeet
tmeas = np.linspace(0.0, t_end, nplot+1)
print(' dtmeet, tmeet', dtmeet, tmeet)
print('tmeas', tmeas)
epsmeet = 10.0**(-10)
nt = int(len(time)/nplot)
print('nt',nt,len(time))
t_plot = time[0::nt]
#print('t_plot', t_plot, nt, nplot, t_end)


##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

#  fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
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
ax4.set_ylabel(r'$H(t)$ ',fontsize=size)


#__________________  Define function spaces  __________________#

nDG = 0
V_W = fd.FunctionSpace(mesh, 'DG', nDG, vfamily='DG') # 
V_C = fd.FunctionSpace(mesh, 'R', 0, vfamily='R', vdegree=0) # billiard variables X, U, Y and V

# Variables for modified midpoint test case billiard
mixed_Vmpc = V_C * V_C * V_C * V_C * V_C
mixed_Vmpc8 = V_C * V_C * V_C * V_C * V_C * V_C * V_C * V_C * V_C

n8 = 0 # use replace to get rid of n+1-level and get 4 variables to solve or do not; solve for 8 variables
if n8==0:
    result_mixedmpc = fd.Function(mixed_Vmpc)
    vvmpc = fd.TestFunction(mixed_Vmpc)
    vvmpc0, vvmpc1, vvmpc2, vvmpc3, vvmpc4 = fd.split(vvmpc) # These represent "blocks".
    Xh12, Uh12, Yh12, Vh12,lamb12 = fd.split(result_mixedmpc)
    X1 = fd.Function(V_C)
    U1 = fd.Function(V_C)
    Y1 = fd.Function(V_C)
    V1 = fd.Function(V_C)
    lam12 = fd.Function(V_C)
else:
    result_mixedmpc8 = fd.Function(mixed_Vmpc8)
    vvmpc8 = fd.TestFunction(mixed_Vmpc8)
    vvmpc0, vvmpc1, vvmpc2, vvmpc3, vvmpc4, vvmpc5, vvmpc6, vvmpc7,vvmpc8 = fd.split(vvmpc8) # These represent "blocks".
    Xh12, Uh12, Yh12, Vh12, lamb12, X1, U1, Y1, V1 = fd.split(result_mixedmpc8)
    X11 = fd.Function(V_C)
    U11 = fd.Function(V_C)
    Y11 = fd.Function(V_C)
    V11 = fd.Function(V_C)
# lamb1, lamb2 = fd.split(result_mixedmpc)

X0 = fd.Function(V_C)
U0 = fd.Function(V_C)
Y0 = fd.Function(V_C)
V0 = fd.Function(V_C)


##_________________  Initial Conditions __________________________##

X0.assign(X00)
U0.assign(U00)
Y0.assign(Y00)
V0.assign(V00)
t = 0.0
i = 0.0
print('t, X00, U00 Y00 V00',t,X00,U00,Y00,V00)
X0p = X00 # np.array(X0.vector())
U0p = U00 # np.array(U0.vector())
Y0p = Y00 # np.array(Y0.vector())
V0p = V00 # np.array(V0.vector())
lam0p = 0.0
ax1.plot(X0p,Y0p,'.')
ax3.plot(t,U0p,'.k')
ax3.plot(t,V0p,'.r')

PEfac = 1.0

q120 = -gamm*(1-(X0p/Lx)**twon-(Y0p/Ly)**twon)-lam0p
if nFmaxfunc==0:
    Fplus0 = 0.5*q120+np.sqrt(bb**2+0.25*q120**2)
elif nFmaxfunc==1:
    Fplus0 = np.where(q120>bb,q120,0.0) + np.where(q120**2<bb**2,(0.25/bb**2)*(q120+bb)**2,0.0) # Fails; Ioannis, C1
elif nFmaxfunc==2:
    Fplus0 = np.where(q120>bb,q120,0.0) + np.where(q120**2<bb**2,(1/(16*bb**3))*(4*bb*(q120+bb)**3-(q120+bb)**4),0.0) # Integrated Ioannis
elif nFmaxfunc==3:
    Fplus0 = bb*np.log(1+np.exp(q120/bb))

E0 = 0.5*(U0p**2+V0p**2) + (0.5/gamm)*(Fplus0**2-lam0p**2) #  - PEfac*(bb**2/gamm)*np.log(1-(X0p/Lx)**twon-(Y0p/Ly)**twon),
# ax2.plot(t,E0,'.')

# 
#
#

# MMP https://epubs.siam.org/doi/10.1137/1.9781611976311
vpolyp = 5
gamm = 100
q12 = -gamm*(1-(Xh12/Lx)**twon-(Yh12/Ly)**twon)-lamb12
# q12 = -gamm*(1-((2*Xh12-X0)/Lx)**twon-((2*Yh12-Y0)/Ly)**twon)-lamb12
bb = 4.0*2.0*0.34*np.sqrt(gamm) # Functions below may each need bespoke value of bb?
if nFmaxfunc ==0:
    bb = 4.0*0.34*np.sqrt(gamm)
    Fplus = 0.5*q12+fd.sqrt(bb**2+0.25*q12**2)
elif nFmaxfunc ==1:
    bb = (2.0*0.34*np.sqrt(gamm))**2
    bb = 1/np.sqrt(gamm)**(1.5)
    Fplus = fd.conditional(q12>bb,q12,0.0) + fd.conditional(q12**2<bb**2,(0.25/bb**2)*(q12+bb)**2,0.0) # Fails often; Ioannis, C1 polynomial
elif nFmaxfunc ==2:
    bb = 1.0
    Fplus = fd.conditional(q12>bb,(q12+bb**2/3)**(0.5),0.0) + fd.conditional(q12**2<bb**2,((q12+bb)**3/(6*bb))**(0.5),0.0) # Fails; integrated F*F' = Ioannis
elif nFmaxfunc ==3: 
    Fplus = bb*fd.ln(1+fd.exp(q12/bb)) # ML one


# VPnl = (1/Lx)*( Uh12*(X1-X0)/dt-Xh12*(U1-U0)/dt+Vh12*(Y1-Y0)/dt-Yh12*(V1-V0)/dt-0.5*(Uh12**2 + Vh12**2) - (0.5/gamm)*(Fplus**2-lamb12**2) )*fd.dx(degree=vpolyp) # next one is times dt for better conditioning
VPnl = (1/Lx)*( Uh12*(X1-X0)-Xh12*(U1-U0)+Vh12*(Y1-Y0)-Yh12*(V1-V0)-0.5*dt*(Uh12**2 + Vh12**2) - dt*(0.5/gamm)*(Fplus**2-lamb12**2) )*fd.dx(degree=vpolyp)

solver_parameters6 = {
    'mat_type': 'nest',
    'snes_type': 'newtonrt',
    "ksp_type": "minres",  # MINRES is better for saddle-point problems
    "pc_type": "fieldsplit",  # Use block preconditioning
    "pc_fieldsplit_type": "schur",  # Schur complement for coupled system
    "pc_fieldsplit_schur_factorization_type": "full",  # Full factorization improves stability
    "pc_fieldsplit_schur_precondition": "selfp",  # Approximate Schur complement

    # Explicitly define the five fields:
    "pc_fieldsplit_0_fields": "0,1,2,3",  # First four fields: primal variables
    "pc_fieldsplit_1_fields": "4",        # Fifth field: Lagrange multiplier

    # Solver for the first four (primal) fields
    "fieldsplit_0_ksp_type": "gmres",  # GMRES for general stability
    "fieldsplit_0_pc_type": "hypre",   # Algebraic multigrid (AMG) for efficient solve

    # Solver for the Lagrange multiplier field
    "fieldsplit_1_ksp_type": "cg",      # Conjugate gradient (good for symmetric problems)
    "fieldsplit_1_pc_type": "jacobi",   # Simple diagonal preconditioner

    # Improve numerical stability for direct solves
    "mat_mumps_icntl_14": 100,  # Better pivoting in direct solver
    "ksp_atol": 1e-8,           # Tight absolute tolerance
    "ksp_rtol": 1e-6,           # Relative tolerance
    "snes_rtol": 1e-6 
    # 'snes_type': 'fas'
    # 'snes_type': 'vinewtonrsls',
    # 'snes_type': 'newtonls',
    # 'snes_atol': 1e-19,
    # 'ksp_type': 'preonly'
    #'snes_monitor_true_residual': None,
    #'snes_view': None
}

solver_parameters7 = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu"
}

solver_parameters9 = {
    "mat_type": "nest",

    "pc_type": "fieldsplit",  # Use block preconditioning

    # Explicitly define the five fields:
    "pc_fieldsplit_0_fields": "0,1,2,3",  # First four fields: primal variables
    "pc_fieldsplit_1_fields": "4",        # Fifth field: Lagrange multiplier
    'pc_fieldsplit_1_snes_type': 'vinewtonssls'
}

solver_parameters19 = {
    "mat_type": "nest",

    # Explicitly define the five fields:
    "pc_fieldsplit_0_fields": "0,1,2,3",  # First four fields: primal variables
    "pc_fieldsplit_1_fields": "4",        # Fifth field: Lagrange multiplier

    # Solver for the Lagrange multiplier field
    # "fieldsplit_1_ksp_type": "cg",      # Conjugate gradient (good for symmetric problems)
    "fieldsplit_1_pc_type": "jacobi"   # Simple diagonal preconditioner    
}

solver_parameters8 = {
    "mat_type": "nest",

    # Explicitly define the five fields:
    "pc_fieldsplit_0_fields": "0,1,2,3",  # First four fields: primal variables
    "pc_fieldsplit_1_fields": "4",        # Fifth field: Lagrange multiplier

    # Solver for the Lagrange multiplier field
    # "fieldsplit_1_ksp_type": "cg",      # Conjugate gradient (good for symmetric problems)
    "fieldsplit_1_pc_type": "jacobi",   # Simple diagonal preconditioner    
}
    
X_expr = fd.derivative(VPnl, Uh12, du=vvmpc1) # du=v_C eqn for X1
U_expr = fd.derivative(VPnl, Xh12, du=vvmpc0) # du=v_C eqn for U1
Y_expr = fd.derivative(VPnl, Vh12, du=vvmpc3) # du=v_C eqn for Y1
V_expr = fd.derivative(VPnl, Yh12, du=vvmpc2) # du=v_C eqn for V1
lamb_expr = fd.derivative(VPnl, lamb12, du=vvmpc4) # du=v_C eqn for lamb
lamb_expr = (gamm/dt)*lamb_expr # scaling
if n8==0:
    X_expr = fd.replace(X_expr, {X1: 2*Xh12-X0}) # X1 = 2*Xh12-X0 
    U_expr = fd.replace(U_expr, {U1: 2*Uh12-U0}) # U1 = 2*Uh12-U0
    Y_expr = fd.replace(Y_expr, {Y1: 2*Yh12-Y0}) # Y1 = 2*Yh12-Y0 
    V_expr = fd.replace(V_expr, {V1: 2*Vh12-V0}) # V1 = 2*Vh12-V0 
    Fexpr = X_expr+U_expr+Y_expr+V_expr+lamb_expr
    #
    solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc), solver_parameters=solver_parameters9)
    # solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc))
else:
    X1_expr = (1/Lx)*(vvmpc4*( X1-2*Xh12+X0 )  )*fd.dx(degree=vpolyp)
    U1_expr = (1/Lx)*(vvmpc5*( U1-2*Uh12+U0 )  )*fd.dx(degree=vpolyp)
    Y1_expr = (1/Lx)*(vvmpc6*( Y1-2*Yh12+Y0 )  )*fd.dx(degree=vpolyp)
    V1_expr = (1/Lx)*(vvmpc7*( V1-2*Vh12+V0 )  )*fd.dx(degree=vpolyp)
    Fexpr = X_expr+U_expr+Y_expr+V_expr+X1_expr+U1_expr+Y1_expr+V1_expr+lamb_expr
    # solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc8), solver_parameters=solver_parameters6)
    solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc8))

#
#    
#  


###### OUTPUT FILES #########
# outfile_Z = fd.File("resultbuoy/billZ.pvd")
# outfile_W = fd.File("resultbuoy/billU.pvd")
# outfile_lamb = fd.File("resultbuoy/billlamb.pvd")
# Exact outline squircles
NXX = 1000
XXp = np.linspace(-Lx,Lx,NXX)
YYpu = Ly*(1-(XXp/Lx)**twon)**(1/twon)
YYpm = -Ly*(1-(XXp/Lx)**twon)**(1/twon)


print('Time Loop starts')
tic = tijd.time()
while t <= 1.0*(t_end + dt): #

    solvelamb_nl.solve()
    if n8==0:
        Xh12, Uh12, Yh12, Vh12, lamb12 = fd.split(result_mixedmpc)
        X1.interpolate(2.0*Xh12-X0)
        Y1.interpolate(2.0*Yh12-Y0)
        U1.interpolate(2.0*Uh12-U0)
        V1.interpolate(2.0*Vh12-V0)
        lam12.interpolate(1.0*lamb12)
        X0p = np.array(X1.vector())
        U0p = np.array(U1.vector())
        Y0p = np.array(Y1.vector())
        V0p = np.array(V1.vector())
        X0p0 = np.array(X0.vector())
        U0p0 = np.array(U0.vector())
        Y0p0 = np.array(Y0.vector())
        V0p0 = np.array(V0.vector())
        lam0p = np.array(lam12.vector())
    else:
        Xh12, Uh12, Yh12, Vh12, X1, U1, Y1, V1, lamb12 = fd.split(result_mixedmpc8)
        X11.interpolate(X1)
        U11.interpolate(U1)
        Y11.interpolate(Y1)
        V11.interpolate(V1)
        X0p = np.array(X11.vector())
        U0p = np.array(U11.vector())
        Y0p = np.array(Y11.vector())
        V0p = np.array(V11.vector())

    t+= dt
    tmeet = tmeet+dtmeet

    if (t in t_plot): #
        ax1.plot(X0p,Y0p,'.k')
        ax3.plot(t,U0p,'.k')
        ax3.plot(t,V0p,'.r')
        q120 = -gamm*(1-(X0p/Lx)**twon-(Y0p/Ly)**twon)-lam0p
        # q120 = -gamm*(1-((2*X0p-X0p0)/Lx)**twon-((2*Y0p-Y0p0)/Ly)**twon)-lam0p
        if nFmaxfunc==0:
            Fplus0 = 0.5*q120+np.sqrt(bb**2+0.25*q120**2)
        elif nFmaxfunc==1:
            Fplus0 = np.where(q120>bb,q120,0.0) + np.where(q120**2<bb**2,(0.25/bb**2)*(q120+bb)**2,0.0) # Fails; Ioannis, C1
        elif nFmaxfunc==2:
            Fplus0 = np.where(q120>bb,q120,0.0) + np.where(q120**2<bb**2,(1/(16*bb**3))*(4*bb*(q120+bb)**3-(q120+bb)**4),0.0) # Integrated Ioannis
        elif nFmaxfunc==3:
            Fplus0 = bb*np.log(1+np.exp(bb*q120))
        E1 = 0.5*(U0p**2+V0p**2) + (0.5/gamm)*(Fplus0**2-lam0p**2) #  - PEfac*(bb**2/gamm)*np.log(1-(X0p/Lx)**twon-(Y0p/Ly)**twon),
        ax2.plot(t, (E1-E0)/E0, '.k')
        # ax2_right = ax2.twinx()
        # ax2_right.plot(t, E1, '.b', label="E1")
        ax4.plot(t, E1, '.k')
        ax1.plot(XXp,YYpu,'-b')
        ax1.plot(XXp,YYpm,'-b')
        # Optional: Improve visibility
        #ax2_right.tick_params(axis='y', colors='b')
        # ax2_right.spines['right'].set_color('b')  # Make right y-axis stand out
        plt.tight_layout()
        #  plt.show()
        plt.pause(0.1)
        print('Plotting starts, E1: ', E1, (E1-E0)/E0)

    if n8==0:
        X0.assign(X1)
        U0.assign(U1)
        Y0.assign(Y1)
        V0.assign(V1)
    else:
        X0.interpolate(X1)
        U0.interpolate(U1)
        Y0.interpolate(Y1)
        V0.interpolate(V1)

    # print('t =', t, tmeet, i) #
            
# End while time loop      

toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
# print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
