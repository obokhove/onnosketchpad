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
nx = 1    # 
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
    CFL = 0.25*0.25 # 0.5*0.125 # 0.5*0.125 # 1.0, 0.5, 0.25, 0.125 
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
nDG = 0
V_C = fd.FunctionSpace(mesh, 'DG', nDG, vfamily='DG') # 
V_Cc = fd.FunctionSpace(mesh, 'R', 0, vfamily='R', vdegree=0) # buoy variables Z and W
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
    rpow = 1.0
    bb = 0.1*dt**rpow
    bb = 10*dt**rpow
    bb = dt**rpow
    dd = bb
    cc = bb # was cc= gamm
    gammm = 10**10
    aa = 10
    set0 = 0.0
    print('cc',cc)
    
    sfac = 0*10**(-8)
    Z_expr = ((1/Lx)*vvmpc0*( (Zh-Z0)/dt-0.5*(Wh+W0) ))*fd.dx(degree=vpolyp)

    # The Potential (SSQP) & AVF form
    Q0 = -gammm*Z0-Thet0
    Qh = -gammm*Zh-Thet
    diff = Qh-Q0
    nta = 2
    if nta==0:
        QhgRmax = fd.conditional(fd.lt(Qh,-gammm*cc),-gammm*cc,Qh)
        QhgRmax = fd.conditional( fd.lt(QhgRmax,cc), QhgRmax, cc )
        Q0gRmax = fd.conditional(fd.lt(Q0,-gammm*cc),-gammm*cc,Q0)
        Q0gRmax = fd.conditional( fd.lt(Q0gRmax,cc), Q0gRmax, cc )
        # old: QhRmax = fd.conditional(fd.lt(Qh,cc),cc,Qh)
        # old: Q0Rmax = fd.conditional(fd.lt(Q0,cc),cc,Q0)
        QhgRmax2g = fd.conditional(fd.lt(Qh, -gammm*cc), 0.0,\
                                   fd.conditional(fd.gt(Qh, cc), (cc + gammm*cc)**2, (Qh + gammm*cc)**2))
        Q0gRmax2g = fd.conditional(fd.lt(Q0, -gammm*cc), 0.0,\
                                   fd.conditional(fd.gt(Q0, cc), (cc + gammm*cc)**2, (Q0 + gammm*cc)**2))
        QhRmax2 = fd.conditional(fd.lt(Qh, cc), cc**2, Qh**2)
        Q0Rmax2 = fd.conditional(fd.lt(Q0, cc), cc**2, Q0**2)
        safediff = fd.conditional(fd.eq(diff, 0.0), 1.0, diff)
        # old: forcelimit = 0.5*( ( (QhgRmax+gammm*cc)**2-(Q0gRmax+gammm*cc)**2 )/(gammm+1)+ QhRmax**2 - Q0Rmax**2 )/diff
        forcelimit = 0.5*( (QhgRmax2g - Q0gRmax2g)/(gammm+1)+ QhRmax2 - Q0Rmax2 )/safediff
        # fd.conditional(fd.gt(0.5*(Q0+Qh),cc),0.5*(Q0+Qh),(0.5*(Q0+Qh)+gammm*cc)/(gammm+1))),forcelimit)
        avg_force = fd.conditional(fd.And(Q0 <= -gammm*cc, Qh <= -gammm*cc), 0.0*Q0,\
                                   fd.conditional(fd.And(Q0 >= cc, Qh >= cc), 0.5*(Q0+Qh),\
                                                  fd.conditional(fd.And(fd.And(Q0 >= -gammm*cc, Qh <= cc),
                                                                        fd.And(Q0 >= -gammm*cc, Qh <= cc)),\
                                                                 (0.5*(Q0+Qh)+gammm*cc)/(gammm+1), forcelimit)))
    elif nta==1:
        safediff = fd.conditional(fd.eq(diff, 0.0), 1.0, diff)
        R11 = fd.And(Q0 <= -gammm*cc, Qh <= -gammm*cc) # Region 11
        val11 = 0.0
        R22 = fd.And(Q0 >= cc, Qh >= cc) # Region 22
        val22 = 0.5*(Q0+Qh)     
        R33 = fd.And(fd.And(Q0 >= -gammm*cc, Q0 <= cc), fd.And(Qh >= -gammm*cc, Qh <= cc)) # Region 33
        val33 = (0.5*(Q0+Qh) + gammm*cc)/(gammm+1)
        R12 = fd.And(Q0 < -gammm*cc, fd.And(Qh > -gammm*cc, Qh < cc)) # Region 12
        val12 = 0.5*(Qh + gammm*cc)**2 / ((gammm+1)*diff)
        R21 = fd.And(Qh < -gammm*cc, fd.And(Q0 > -gammm*cc, Q0 < cc)) # Region 21
        val21 = -0.5*(Q0 + gammm*cc)**2 / ((gammm+1)*diff)
        R13 = fd.And(Q0 < -gammm*cc, Qh > cc) # Region 13
        val13 = 0.5*( (Qh**2 - cc**2) + (gammm+1)*cc**2 ) / diff
        R31 = fd.And(Qh < -gammm*cc, Q0 > cc)         # Region 31
        val31 = -0.5*( (Q0**2 - cc**2) + (gammm+1)*cc**2 ) / diff
        R23 = fd.And(fd.And(Q0 > -gammm*cc, Q0 < cc), Qh > cc) # Region 23
        val23 = 0.5*( (Qh**2 - cc**2) + (gammm+1)*cc**2 - (Q0 + gammm*cc)**2/(gammm+1) ) / diff
        R32 = fd.And(fd.And(Qh > -gammm*cc, Qh < cc), Q0 > cc) # Region 32
        val32 = 0.5*( (Qh + gammm*cc)**2/(gammm+1) - (Q0**2 - cc**2) - (gammm+1)*cc**2 ) / diff
        # Nested conditionals avg_force = fd.conditional(R11, val11, fd.conditional(R22, val22, fd.conditional(R33, val33,\fd.conditional(R12, val12, fd.conditional(R21, val21, fd.conditional(R13, val13,\ fd.conditional(R31, val31, fd.conditional(R23, val23, fd.conditional(R32, val32, 0.0)))))))))
        avg_force = fd.conditional(R11, val11, 0.0) + fd.conditional(R22, val22, 0.0) + \
            fd.conditional(R33, val33, 0.0) + fd.conditional(R12, val12, 0.0) + \
            fd.conditional(R21, val21, 0.0) + fd.conditional(R13, val13, 0.0) + \
            fd.conditional(R31, val31, 0.0) + fd.conditional(R23, val23, 0.0) + \
            fd.conditional(R32, val32, 0.0)
    elif nta==2:
        R11 = fd.And(Q0 <= 0.0, Qh <= 0.0) # Region 11
        val11 = 0.0
        R22 = fd.And(Q0 >= 0.0, Qh >= 0.0) # Region 22
        val22 = 0.5*(Q0+Qh)
        R12 = fd.And(Q0 < 0.0, Qh > 0.0) # Region 12 excluding borders
        val12 = 0.5*Qh**2 / diff
        R21 = fd.And(Qh < 0.0, Q0 > 0.0) # Region 21 escluding borders
        val21 = -0.5*Q0**2 / diff
        avg_force = fd.conditional(R11,val11,0.0)+fd.conditional(R22,val22,0.0)+fd.conditional(R12,val12,0.0)+fd.conditional(R21,val21,0.0)   
        
    RHSW = -1 + avg_force
    
    W_expr = ((1/Lx)*vvmpc1*( (Wh-W0)/dt-RHSW ))*fd.dx(degree=vpolyp)
    T_expr = ((1/Lx)*vvmpc2*( 0.5*(Thet+Thet0) + avg_force ))*fd.dx(degree=vpolyp)
    Fexpr = Z_expr+W_expr+T_expr

    # Define trial function for Jacobian
    du = fd.TrialFunction(result_mixedmpc.function_space())
    dZ, dW, dThet_t = fd.split(du)
    nuu = 0.5*gammm/bb
    deltaa = Zh-Z0
    nudelt = nuu*deltaa
    J11 = (1/Lx)*vvmpc0*(dZ/dt)*fd.dx(degree=vpolyp)
    J12 = (1/Lx)*vvmpc0*(-0.5*dW)*fd.dx(degree=vpolyp)
    J22 = (1/Lx)*vvmpc1*(dW/dt)*fd.dx(degree=vpolyp)
    J21 = (1/Lx)*vvmpc1*(-dZ*aa*nuu*fd.exp(-nuu*(Zh+Z0))*fd.conditional(fd.eq(deltaa,0.0),-1.0,\
        (fd.cosh(nudelt)/nudelt-fd.sinh(nudelt)/nudelt-fd.sinh(nudelt)/nudelt**2) ) )*fd.dx(degree=vpolyp)
    J31 = (1/Lx)*vvmpc2*( 1.0*dZ)*fd.dx(degree=vpolyp)
    J33 = (1/Lx)*vvmpc2*(-1.0*dThet_t)*fd.dx(degree=vpolyp)
    Jexpr = J11+J12+J21+J22+J31+J33
    
    solver_parameters = {
        'mat_type': 'nest',
        "snes_type": "vinewtonrsls",  # added "vinewtonrsls" works for dt=0.035 rest fails at 0.09; defualt too but works at 0.0075
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": 1e-16, #  was 1.e-16
        "snes_rtol": 1e-10,
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
    }

    solver_parameters16 = {
        #"snes_type": "newtonls",  # added "vinewtonrsls" or "newtontr" fails
        "snes_type": "vinewtonrsls",  # added "vinewtonrsls" works for dt=0.035 rest fails at 0.09; defualt too but works at 0.0075; default solver fails
        #"snes_monitor": None,
        #"snes_converged_reason": None, "snes_linesearch_type": "l2",   # Take full steps across the non-smooth interface
        #"snes_divergence_tolerance": 1.e10,
        "snes_stol": 1e-16, 
        "snes_atol": 1e-16, #  was 1.e-16
        "snes_rtol": 1e-12, # works with 10**-10 fails with 10**-16
        "ksp_rtol": 1e-14, # works with 10**-10,-14
        "ksp_atol": 1e-14, # works woth 10**-12,-14
    }

    solver_parameters18 = {
        'snes_type': 'newtonls',
        'snes_linesearch_type': 'l2',  # 'basic' or 'l2'
        'snes_max_it': 50,
        #'snes_monitor': True,
        'ksp_type': 'preonly',
        'pc_type': 'lu'
    }
    
    
    # solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc),solver_parameters=solver_parameters,options_prefix="softball_",nullspace=None,pre_jacobian_callback=None,post_jacobian_callback=None,snes_error_if_not_converged=False) # fails
    nJJ = 0
    if nJJ==1:
        solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr,result_mixedmpc,J=Jexpr), solver_parameters=solver_parameters)  # fails, needs nest
    else:
        lbound = fd.Function(mixed_Vmpc)
        ubound = fd.Function(mixed_Vmpc)
        INF_VAL = 1e20
        # --- Block 0: Zh >= 0 ---
        lbound.sub(0).interpolate(fd.Constant(0.0))
        ubound.sub(0).interpolate(fd.Constant(INF_VAL))
        # --- Block 1: Wh is completely unconstrained ---
        lbound.sub(1).interpolate(fd.Constant(-INF_VAL))
        ubound.sub(1).interpolate(fd.Constant(INF_VAL))
        # --- Block 2: Thet (lambda) <= 0 ---
        lbound.sub(2).interpolate(fd.Constant(-INF_VAL))
        ubound.sub(2).interpolate(fd.Constant(0.0))
        solvelamb_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr,result_mixedmpc),solver_parameters=solver_parameters16) 
# End if

###### OUTPUT FILES ########## outfile_Z = fd.VTKFile("resultbuoy/Z.pvd") outfile_W = fd.VTKFile("resultbuoy/W.pvd")

print('Time Loop starts')
tic = tijd.time()
while t <= 1.0*(t_end - dt): #
    if nvpcase == 2:
        #result_mixedmpc.sub(0).assign(Z0)
        #result_mixedmpc.sub(1).assign(W0)
        # result_mixedmpc.sub(2).assign(Thet0) solvelamb_nl.solve(bounds=(lbound, ubound))
        #
        solvelamb_nl.solve()
        Zh, Wh, Thet = fd.split(result_mixedmpc)
        W1 = fd.assemble(interpolate(Wh,V_C))
        Z1 = fd.assemble(interpolate(Zh,V_C))
        Thet1 = fd.assemble(interpolate(Thet,V_C))
        # Z1.assign(Zh)
        # W1.assign(Wh)
    # End if
    
    t+= dt
    print('t',t)
    # if (t in t_plot): # 
    # print('Plotting starts')
    #     i += 1
    tmeet = tmeet+dtmeet
    if nvpcase==2:
        Z00 = np.array(Z0.vector())
        W00 = np.array(W0.vector())
        Thet00 = np.array(Thet0.vector())
        Z11 = np.array(Z1.vector())
        W11 = np.array(W1.vector())
        Thet11 = np.array(Thet1.vector())
        ax1.plot([t-dt,t],[Z00,Z11],'-k')
        Q00 = -gammm*Z00-Thet00
        Qhh = -gammm*Z11-Thet11
        if nta==1:
            Fplus00 = np.where(Q00 > cc, Q00, np.where(Q00 < -gammm*cc, 0.0, (Q00 + gammm*cc)/(gammm + 1.0)))
            Fplus11 = np.where(Qhh > cc, Qhh, np.where(Qhh < -gammm*cc, 0.0, (Qhh + gammm*cc)/(gammm + 1.0)))
            Fplus00 = np.where(Q00 > cc, np.sqrt(Q00**2+gammm*cc**2), np.where(Q00 < -gammm*cc, 0.0, (Q00 + gammm*cc)/np.sqrt(gammm + 1.0)))
            Fplus11 = np.where(Qhh > cc, np.sqrt(Qhh**2+gammm*cc**2), np.where(Qhh < -gammm*cc, 0.0, (Qhh + gammm*cc)/np.sqrt(gammm + 1.0)))
            H00 = 0.5*W00**2+Z00+0.5*(Fplus00**2 - np.where(Q00 > cc, gammm * cc**2, 0.0)  - Thet00**2)/gammm # 07-06-2026
            H11 = 0.5*W11**2+Z11+0.5*(Fplus11**2 - np.where(Qhh > cc, gammm * cc**2, 0.0)  - Thet11**2)/gammm # 07-06-2026
            H00 = 0.5*W00**2+Z00+0.5*(Fplus00**2 - Thet00**2)/gammm # 07-06-2026
            H11 = 0.5*W11**2+Z11+0.5*(Fplus11**2 - Thet11**2)/gammm # 07-06-2026
        elif nta==2:
            Fplus00 = np.where(Q00 > 0.0, Q00, 0.0)
            Fplus11 = np.where(Qhh > 0.0, Qhh, 0.0)
            H00 = 0.5*W00**2+Z00+0.5*(Fplus00**2 - Thet00**2)/gammm # 14-06-2026
            H11 = 0.5*W11**2+Z11+0.5*(Fplus11**2 - Thet11**2)/gammm # 14-06-2026
        ax2.plot([t-dt,t],[H00,H11],'-k')
        # ax3.plot([Z00,Z11],[W00,W11],'-k')
        ax3.plot(Z11,W11,'.k')
        if (Q00<-gammm*cc and Qhh > cc) or (Qhh<-gammm*cc and Q00 > cc):
            print('13 or 31 exchange; beware!')

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
plt.savefig("figs/acwaveEcollball2026ftaalamnew.png")

tmmax = t # 7*tmax
Wexnd = -wmax-(tmmax-5*tmax)
Zexnd = -0.5*(tmmax-5*tmax)**2-wmax*(tmmax-5*tmax)
ax1.plot(tmmax,Zexnd,'xr')
ax1.plot(tmmax,Znn,'ob')
Linferror = np.abs(Wexnd-Wnn)+np.abs(Znn-Zexnd)
L2error = np.sqrt((Wexnd-Wnn)**2+(Znn-Zexnd)**2)
print(f"gammm: {gammm:.5f}, cc: {cc:.15f}, dt: {dt:.5f}, Linferror: {Linferror:.5f}, L2error: {L2error:.5f}")
print('shit',Znn,Zexnd,Wnn,Wexnd,wmax,7*tmax, t_end, t)

toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
# print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
