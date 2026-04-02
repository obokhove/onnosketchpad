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
from firedrake import VTKFile #  ONNO added extra
from firedrake.__future__ import interpolate
import os.path
# import os.path
# Run-time instructions: . /Users/amtob/firedrake/bin/activate or . /Users/onnobokhove/amtob/werk/firedrake/bin/activate
# Docker install app; then run: 
#
# Program energy-conserving softened lambda-eliminated test case ball
#
# parameters in SI units
# 
Lx = 10.0 # 
nx = 2    # 
nCG = 1   # function space order horizontal

#__________________  FIGURE PARAMETERS  _____________________#
tsize2 = 12 # font size of image axes
size = 16   # font size of image axes
t = 0
tt = format(t, '.3f') 

#________________________ MESH  _______________________#
# 
mesh = fd.IntervalMesh(nx, Lx)
x = fd.SpatialCoordinate(mesh)
xvals = np.linspace(0.0, Lx-10**(-10), nx)
xslice = 0.5*Lx
## initial condition nic=1
x = mesh.coordinates
t0 = 0.0
nic = 0
nvpcase = 2 # Energy-conserving lambda eliminated: 2
if nvpcase == 2: # 
    nic = 1
time = []
t = 0  
if nic == 1:
    Zc = 5.0
    Wc = 0.0
    tmax = Wc+np.sqrt(2*Zc-4*Wc**2)
    Tperiod = np.sqrt(10)
    Tperiod = tmax
    nTfac = 7
    t_end = nTfac*Tperiod # time of simulation [s]
    Tstartmeas = 0.0
    dtt = np.minimum(0.018,0.018) # i.e.
    dtt = np.minimum(0.02,0.02) # i.e. 
    Nt = 1 # 
    CFL = 0.5*0.125 # 1.0, 0.5, 0.25, 0.125 
    dt = CFL*dtt # CFL*dtt
    print('dtt=',dtt, t_end/dtt)      
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 200
    Thetac = np.log(Zc)
    
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
fig, (ax1, ax2, ax3) = plt.subplots(3)
# 
ax1.set_title(r'VP bouncing ball $Z>0$: Energy-conserving Firedrake',fontsize=tsize2)   
ax1.set_ylabel(r'$Z(t)$ ',fontsize=size)
ax1.set_xlabel(r'$t$ ',fontsize=size)
ax3.set_ylabel(r'$W(t)$ ',fontsize=size)
ax3.set_xlabel(r'$Z(t)$ ',fontsize=size)
ax2.set_ylabel(r'$H(t)$ ',fontsize=size)
ax2.set_xlabel(r'$t$ ',fontsize=size)

#__________________  Define function spaces  __________________#

V_W = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='CG') # 
V_C = fd.FunctionSpace(mesh, 'R', 0, vfamily='R', vdegree=0) # buoy variables Z and W
# Variables for energy-conserving softened lambda-eliminated test case ball
mixed_Vmpc = V_C * V_C *V_C
result_mixedmpc = fd.Function(mixed_Vmpc)
vvmpc = fd.TestFunction(mixed_Vmpc)
vvmpc0, vvmpc1, vvmpc2 = fd.split(vvmpc)  # These represent "blocks".
Zh, Wh, Thet = fd.split(result_mixedmpc)
Z0 = fd.Function(V_C)
Z1 = fd.Function(V_C)
W0 = fd.Function(V_C)
W1 = fd.Function(V_C)
Thet0 = fd.Function(V_C)
Thet1 = fd.Function(V_C)

##_________________  Initial Conditions __________________________##
Z0.assign(Zc)
W0.assign(Wc)
Thet0.assign(Thetac)
t = 0.0
i = 0.0
print('t, Zc, Wc',t,Zc,Wc)
Z00 = np.array(Z0.vector())
W00 = np.array(W0.vector())
ax1.plot(t,Z00,'.')
# ax2.plot(t,W00,'.')
#
if nvpcase==2: # Energy conserving softened potential lamb eliminated; weak formulations
    vpolyp = 5
    gamm = 0.25*50
    bb = 2.0*0.34*np.sqrt(gamm)
    bb = 10*dt**(3/2)
    cc = 1/bb # was cc= gamm
    gammm = 4
    aa = 10
    
    sfac = 0*10**(-8)
    Z_expr = ((1/Lx)*vvmpc0*( (Zh-Z0)/dt-0.5*(Wh+W0) ))*fd.dx(degree=vpolyp)

    # RHSW = fd.conditional( fd.eq(Zh,Z0) , -1+(1/cc)*fd.exp(-0.5*cc*(Zh+Z0)), -1-(1/cc**2)*(fd.exp(-cc*Zh)-fd.exp(-cc*Z0))/(Zh-Z0) )
    RHSW = fd.conditional( fd.eq(Zh,Z0) , -1+aa*fd.exp(-0.5*cc*gammm*(Zh+Z0)), -1-(aa/(gammm*cc))*(fd.exp(-cc*gammm*Zh)-fd.exp(-gammm*cc*Z0))/(Zh-Z0) ) # 04-02-2026
    W_expr = ((1/Lx)*vvmpc1*( (Wh-W0)/dt-RHSW ))*fd.dx(degree=vpolyp)

    T_expr = ((1/Lx)*vvmpc2*( Zh-Thet ))*fd.dx(degree=vpolyp)
    Fexpr = Z_expr+W_expr+T_expr
    solver_parameters2 = {
        'mat_type': 'nest',
        'pc_type': 'none',   # Use block Jacobi preconditioner for the second block
        'ksp_max_it': 100,   # Maximum number of iterations for the linear solve
        'snes_max_it': 100,  # Maximum number of Picard iterations
        'ksp_rtol': 1e-12,   # Relative tolerance for linear solve
    }
    solver_parameters2 = {
        'mat_type': 'nest',
        # "snes_type": "newtonls",  # added "vinewtonrsls" or "newtontr"
        # 'snes_linesearch_type': 'bt',  # backtracking
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": 1e-16,
        "snes_rtol": 1e-10,
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
    }
    solver_parameters = {
        'mat_type': 'nest',
        "snes_type": "vinewtonrsls",  # added "vinewtonrsls" works for dt=0.035 rest fails at 0.09; defualt too but works at 0.0075
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": 1e-16,
        "snes_rtol": 1e-10,
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
    }
    solver_parameters11 = {
        "mat_type": "aij",  
        "snes_type": "newtontr",        # Trust-region Newton (LM-like)
        "ksp_type": "gmres",            # or "preonly" + "lu"
        "pc_type": "lu",   # or other preconditioner
        "mat_factor_shift_type": "NONZERO",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": 1e-16,
        "snes_rtol": 1e-10,
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
    }
    
    
    # solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc),solver_parameters=solver_parameters,options_prefix="softball_",nullspace=None,pre_jacobian_callback=None,post_jacobian_callback=None,snes_error_if_not_converged=False) # fails
    solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc), solver_parameters=solver_parameters)  # fails, needs nest
# End if

###### OUTPUT FILES ########## outfile_Z = fd.VTKFile("resultbuoy/Z.pvd") outfile_W = fd.VTKFile("resultbuoy/W.pvd")

print('Time Loop starts')
tic = tijd.time()
while t <= 1.0*(t_end - dt): #
    if nvpcase == 2:
        solvelamb_nl.solve()
        Zh, Wh, Thet = fd.split(result_mixedmpc)
        W1 = fd.assemble(interpolate(Wh,V_C))
        Z1 = fd.assemble(interpolate(Zh,V_C))
        Thet1 = fd.assemble(interpolate(Thet,V_C))
        # Z1.assign(Zh)
        # W1.assign(Wh)
    # End if
    
    t+= dt
    # if (t in t_plot): # 
    # print('Plotting starts')
    #     i += 1
    tmeet = tmeet+dtmeet
    if nvpcase==2:
        Z00 = np.array(Z0.vector())
        W00 = np.array(W0.vector())
        Z11 = np.array(Z1.vector())
        W11 = np.array(W1.vector())
        ax1.plot([t-dt,t],[Z00,Z11],'-k')
        # H00 = 0.5*W00**2+Z00+(1/cc**2)*np.exp(-cc*Z00) # 
        # H11 = 0.5*W11**2+Z11+(1/cc**2)*np.exp(-cc*Z11) # 
        H00 = 0.5*W00**2+Z00+(aa/(gammm*cc))*np.exp(-cc*gammm*Z00) # 04-02-2026
        H11 = 0.5*W11**2+Z11+(aa/(gammm*cc))*np.exp(-cc*gammm*Z11) # 04-02-2026
        ax2.plot([t-dt,t],[H00,H11],'-k')
        #ax3.plot([Z00,Z11],[W00,W11],'-k')
        ax3.plot(Z11,W11,'.k')
        
    Z0.assign(Z1)
    W0.assign(W1)
    Thet0.assign(Thet1)
    # End if
        
# End while time loop      

# Exact phase portrait
Znn = Z00[0]
Wnn = W00[0]
W0 = 0
Z0 = 5
H0 = 0.5*W0+Z0
Nzz = 5000
zmin = 0.0
zmax = H0
zz = np.linspace(zmin, zmax, Nzz)
ww = np.sqrt(2*(H0-zz))
ax3.plot(zz,ww,'--k')
ax3.plot(zz,-ww,'--k')
tmin = 0.0
tmax = Wc+np.sqrt(2*Zc-4*Wc**2)
ttt = np.linspace(tmin, tmax, Nzz)
ax1.plot(ttt,-0.5*ttt**2+Wc*ttt+Zc,'-r')
tmax2 = 3*tmax
wmax = Wc-tmax
tttt = np.linspace(tmax, tmax2, Nzz)
ax1.plot(tttt,-0.5*(tttt-tmax)**2-wmax*(tttt-tmax),'-r')
tttt = np.linspace(3*tmax, 5*tmax, Nzz)
ax1.plot(tttt,-0.5*(tttt-3*tmax)**2-wmax*(tttt-3*tmax),'-r')
tttt = np.linspace(5*tmax, 7*tmax, Nzz)
ax1.plot(tttt,-0.5*(tttt-5*tmax)**2-wmax*(tttt-5*tmax),'-r')
plt.savefig("figs/acwaveEcollball2026ftaa.png")

tmmax = t # 7*tmax
Wexnd = -wmax-(tmmax-5*tmax)
Zexnd = -0.5*(tmmax-5*tmax)**2-wmax*(tmmax-5*tmax)
ax1.plot(tmmax,Zexnd,'xr')
ax1.plot(tmmax,Znn,'ob')
Linferror = np.abs(Wexnd-Wnn)+np.abs(Znn-Zexnd)
L2error = np.sqrt((Wexnd-Wnn)**2+(Znn-Zexnd)**2)
print(f"gam: {gammm:.2f}, a: {aa:.5f}, b: {1/cc:.5f}, dt: {dt:.5f}, Linferror: {Linferror:.5f}, L2error: {L2error:.5f}")
print('shit',Znn,Zexnd,Wnn,Wexnd,wmax,7*tmax, t_end, t)

toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
# print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
