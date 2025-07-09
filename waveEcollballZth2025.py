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
import ufl
from firedrake import *
# from ufl import absolute_value
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
    Tperiod = np.sqrt(10)
    nTfac = 3.5
    t_end = nTfac*Tperiod # time of simulation [s]
    Tstartmeas = 0.0
    dtt = np.minimum(0.01,0.0005) # i.e. 
    Nt = 1 # 
    CFL = 0.5
    dt = CFL*dtt # CFL*dtt
    print('dtt=',dtt, t_end/dtt)      
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 400
    Zc = 5.0
    thetac = np.log(Zc)
    Wc = 0.0
    
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
ax1.set_title(r'VP bouncing ball $Z>0$: Energy-conserving Firedrake, $\theta$ used:',fontsize=tsize2)   
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
mixed_Vmpc = V_C * V_C
result_mixedmpc = fd.Function(mixed_Vmpc)
vvmpc = fd.TestFunction(mixed_Vmpc)
vvmpc0, vvmpc1 = fd.split(vvmpc)  # These represent "blocks".
Zh, Wh = fd.split(result_mixedmpc)
Z0 = fd.Function(V_C)
Z1 = fd.Function(V_C)
W0 = fd.Function(V_C)
W1 = fd.Function(V_C)

##_________________  Initial Conditions __________________________##
Z0.assign(thetac)
W0.assign(Wc)
t = 0.0
i = 0.0
print('t, Zc, Wc',t,Zc,Wc)
Z00 = np.array(Z0.vector())
W00 = np.array(W0.vector())
ax1.plot(t,np.exp(Z00),'.')
# ax2.plot(t,W00,'.')
#
if nvpcase==2: # Energy conserving softened potential lamb eliminated; weak formulations
    vpolyp = 5
    gamm = 100
    bb = 2.0*0.34*np.sqrt(gamm)
    RHStheta = fd.conditional(abs(Zh-Z0)<10**(-10), fd.exp(-Z0)*0.5*(Wh+W0) ,(W0*fd.exp(-Z0)-Wh*fd.exp(-Zh))/(Zh-Z0) + (Wh-W0)*(fd.exp(-Z0)-fd.exp(-Zh))/(Zh-Z0)**2 )
    Z_expr = ((1/Lx)*vvmpc0*( (Zh-Z0)/dt-RHStheta                        ))*fd.dx(degree=vpolyp)
    RHSW = fd.conditional(abs(Zh-Z0)<10**(-10), -1+(bb**2/gamm)*fd.exp(-Z0), -1+ (bb**2/gamm)*fd.exp(-Z0)*(1-fd.exp(-(Zh-Z0)))/(Zh-Z0) )
    W_expr = ((1/Lx)*vvmpc1*( (Wh-W0)/dt-RHSW ))*fd.dx(degree=vpolyp)
    Fexpr = Z_expr+W_expr
    
    solver_parameters2 = {
        'mat_type': 'nest',
        'pc_type': 'none',   # Use block Jacobi preconditioner for the second block
        'ksp_max_it': 100,   # Maximum number of iterations for the linear solve
        'snes_max_it': 100,  # Maximum number of Picard iterations
        'ksp_rtol': 1e-12,   # Relative tolerance for linear solve
    }
    solver_parameters = {
        'mat_type': 'nest',
    }     
    # 
    solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc),solver_parameters=solver_parameters)  # fails, needs nest
# End if

###### OUTPUT FILES ########## outfile_Z = fd.VTKFile("resultbuoy/Z.pvd") outfile_W = fd.VTKFile("resultbuoy/W.pvd")

print('Time Loop starts')
tic = tijd.time()
while t <= 1.0*(t_end + dt): #
    if nvpcase == 2:
        solvelamb_nl.solve()
        Zh, Wh = fd.split(result_mixedmpc)
        W1 = fd.assemble(interpolate(Wh,V_C))
        Z1 = fd.assemble(interpolate(Zh,V_C))
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
        ax1.plot([t-dt,t],[np.exp(Z00),np.exp(Z11)],'-k')
        H00 = 0.5*W00**2+np.exp(Z00)-(bb**2/gamm)*(Z00)
        H11 = 0.5*W11**2+np.exp(Z11)-(bb**2/gamm)*(Z11)
        ax2.plot([t-dt,t],[H00,H11],'-k')
        ax3.plot([np.exp(Z00),np.exp(Z11)],[W00,W11],'-k')
        
    Z0.assign(Z1)
    W0.assign(W1)
    # End if
        
# End while time loop      

# Exact phase portrait
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
tmax = Wc+np.sqrt(2*Zc-Wc**2)
ttt = np.linspace(tmin, tmax, Nzz)
ax1.plot(ttt,-0.5*ttt**2+Wc*ttt+Zc,'-r')
tmax2 = 3*tmax
wmax = Wc-tmax
tttt = np.linspace(tmax, tmax2, Nzz)
ax1.plot(tttt,-0.5*(tttt-tmax)**2-wmax*(tttt-tmax),'-r')
plt.savefig("figs/waveEcollballZth2025.png")

toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
# print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
