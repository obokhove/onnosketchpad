#
# 2D horizontal BLE water-wave equations in x-periodic channel based on implemenation with VP
# =================================================
# Onno Bokhove 05-02-2025 
#
# .. rst-class:: emphasis
#
#     Contributed by `Onno Bokhove <mailto:O.Bokhove@leeds.ac.uk>`__.
#
# Time-step choices: AVF energy-conserving approach
#
# Initial conditions/tests: 
#
import firedrake as fd
from firedrake.petsc import PETSc
import math
from math import *
import time as tijd
import numpy as np
import sympy as sp
from sympy import summation
from sympy.abc import k
import matplotlib
nmpi=0
if nmpi==0:
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import os
import os.path
from firedrake import *
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from finat.point_set import PointSet, GaussLegendrePointSet, GaussLobattoLegendrePointSet
from finat.quadrature import QuadratureRule, TensorProductQuadratureRule
from meshing import get_mesh
os.environ["OMP_NUM_THREADS"] = "1"

# op2.init()
# parameters["coffee"]["O2"] = False

# parameters in SI units REMIS error of polynomials

# water domain and discretisation parameters
# nic = "linearw" # choice initial condition
nic = "rest" # rest state start
nvpcase = "AVF" # AVF= VP time discretisation energy-conserving approach

nCG = 1     # order of CG. Choices: ["1","2","4"] 
multiple=1    # a resolution factor in spatial domain.  1 is basic. 2 is dense. 
multiple2=1   # a resolution factor in temporal domain. 1 is basic. 2 is dense.
nprintout = 0 # print out on screen or in file; 1: on screen: 0: PETSC version
     
if nic=="rest": # rest test
    eps = 0.05 # WW2022-paper setting
    muu = eps**2
    muu_pfe=muu
    Lz = 20
    H0 = 0.15
    Hk = 0.04
    nx = 19 # 39
    ny = 101 # 401
    nvpcase = "AVF" #
    Lx = 0.2
    Ly = 2.0
    grav = 9.81
    thetac = np.pi*68.26/180
    Lc = Ly-0.5*Lx*np.tan(thetac)
    rho0 = 997
    Mm = 0.283*2
    alp3 = 0.5389316
    tan_th = (Ly-Lc)/(0.5*Lx)
    Keel = 0.04
    tanalp = np.tan(alp3)
    meff = Mm/rho0
    Zbar = H0+Keel-( 3*Mm*np.tan(thetac)*np.tan(alp3)**3/rho0 )**(1/3)
    
    
    
#________________________ MESH  _______________________#
'''
    Generates a quadrilateral mesh for a tank with a V-shaped contraction.
    
    The y=0 boundary is the wavemaker and has an ID of 1.
    
    An optional argument Lb can be passed, which denotes the position of the
    waterline of a buoy placed in the contraction. If Lb is passed, all nodes
    to the left of the waterline have an ID of 0 (other than the wavemaker,
    which retains its ID of 1), while the waterline itself has an ID of 2.
    
    Parameters
    ----------
    Lx : float or int
        The width of the tank.
    Ly : float or int
        The length of the (entire) tank.
    Nx : int, must be even
        Number of elements in the x direction.
    Ny : int
        Number of elements in the y direction (in the rectangular section of
        the tank).
    d : float or int, optional
        The (diagonal) length of the contraction. Exactly one of d, Lc or Lr
        must be given. The other two values are calculated.
    Lc : float or int, optional
        The (horizontal) length of the contraction. Exactly one of d, Lc or Lr
        must be given. The other two values are calculated.
    Lr : float or int, optional
        The length of the rectangular part. Exactly one of d, Lc or Lr must be
        given. The other two values are calculated.
    Lb : float or int, optional
        The position of the waterline of a buoy placed in the contraction. Not
        passing Lb (or passing None) is the same as saying there is no buoy.
    
    Returns
    -------
    firedrake.mesh.MeshGeometry
        The mesh, to be used in firedrake.
    Lr : float
        The length of the rectangular part.
'''
#
#  quadrilateral=True, reorder=None,distribution_parameters=None,diagonal=None,comm=COMM_WORLD,name='mesh2d') # xx1, xx2, yy1, yy2
# 
Lxx = Lx
Lyy = Ly
nxx = nx
nyy = ny
Lr = (Ly-Lc) # 0.2*Ly
Area = Lx*Ly-0.5*Lx*Lr
print('Lx etc. ... tanalp',Lx,Ly,Lr,Area,tanalp)

mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)[0]
mesh = mesh2d
# x, y, z = fd.SpatialCoordinate(mesh)
plt.figure(1)
fig, ax = plt.subplots(figsize=(4, 16), subplot_kw=dict(xticks=[], yticks=[]))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
#coords = mesh2d.coordinates.dat.data  # Get mesh coordinates
#coords[:, [0, 1]] = coords[:, [1, 0]]  # Swap x and y
triplot(mesh2d, axes=ax)  
plt.xticks(np.linspace(0, Lx, 5))  # Define custom tick locations
plt.yticks(np.linspace(0, Ly, 5))
# Set labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('mesh2d.png', bbox_inches='tight')
nnopen=0
if nnopen==1:
    from PIL import Image
    Image.open('mesh2d.png').rotate(-90, expand=True).save('mesh2d.png')

x1, y = fd.SpatialCoordinate(mesh2d)

x = mesh2d.coordinates

top_id = 'top'
seps = 0.0 #  -10**(-10)
yvals = np.linspace(0.0+seps, Ly-seps, ny)
xslice = 0.5*Lx
# 

t = 0
fac = 1.0 # Used to split h=H0+eta in such in way that we can switch to solving h (fac=0.0) or eta (fac=1.0)
if nic=="rest":    
    ttilde = 50 # time units 
    t_end = 1.8 # 0.8*1.5
    Nt = 10
    dt = t_end/Nt #
    dtt = 1.2/Nt
    dt3 = np.minimum(Lx/nx,ny/Ly)/np.sqrt(grav*H0)
    CFL1 = 0.25*0.125
    CFL = 0.5
    dt = CFL*dt3
    print('time steps',dt,CFL1*dtt,CFL1*dt3,dt3)
    nplot = 10
    tijde = []
    while (t <= t_end+1*dt):
        tijde.append(t)
        t+= dt
    nt = int(len(tijde)/nplot)
    t_plot = tijde[0::nt]
    plt.pause(2)

##_________________  FIGURE SETTINGS __________________________##
                       
#__________________  Quadratures and define function spaces  __________________#

orders = 2*nCG  # horizontal

                                  
V_W = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='CG') # interior potential varphi; can mix degrees in hor and vert
V_R = fd.FunctionSpace(mesh, 'R', 0) # free surface eta and surface potential phi extended uniformly in vertical: vdegree=0

phi0 = fd.Function(V_W, name="phi0") # velocity potential at level n at free surface
h0 = fd.Function(V_W, name="h0") # water depth at level n
q0 = fd.Function(V_W, name="q0") # q at level n
Z0 = fd.Function(V_R, name="Z0") # Z at level n
W0 = fd.Function(V_R, name="W0") # W at level n

phi1 = fd.Function(V_W, name="phi1") # velocity potential at level n+1 at free surface
h1 = fd.Function(V_W, name="h1") # water depth at level n+1
q1 = fd.Function(V_W, name="q1") # q at level n+1
Z1 = fd.Function(V_R, name="Z1") # Z at level n+1
W1 = fd.Function(V_R, name="W1") # W at level n+1

if nvpcase=="AVF":
    # Variables at midpoint for modified midpoint waves
    mixed_Vmp = V_W * V_W * V_W * V_R * V_R
    result_mixedmp = fd.Function(mixed_Vmp)
    vvmp = fd.TestFunction(mixed_Vmp)
    vvmp0, vvmp1, vvmp2, vvmp3, vvmp4 = fd.split(vvmp) # These represent "blocks".
    phin, hn, qn, Zn, Wn= fd.split(result_mixedmp) # 

# Initialise variables; projections on main variables at initial time
vpolyp = 15
cc = fd.Constant(100)
ccinv = cc**(-1)
# VP or weak forms hydrostatic rest case
phi_hyexpr = ( vvmp0*( phin ))*fd.dx(degree=vpolyp) # nil/rest solution
h_hyexpr   = ( vvmp1*( grav*(hn-H0)+ccinv*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) ))*fd.dx(degree=vpolyp)
q_hyexpr   = ( vvmp2*( qn ))*fd.dx(degree=vpolyp)   # nil/rest solution
Z_hyexpr   = ( vvmp3*( Area*Mm*grav-rho0*ccinv*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) ))*fd.dx(degree=vpolyp)
W_hyexpr   = ( vvmp4*( Wn ))*fd.dx(degree=vpolyp)   # nil/rest solution
F_hyexprnl = phi_hyexpr+h_hyexpr+q_hyexpr+Z_hyexpr+W_hyexpr

# VP or weak forms dynamic case: first test is rest state stays rest state; next test add forcing
Amp = 0.0125 #  0.025
L1 = 0.2
Twm = 0.5
sigma = 2.0*np.pi/Twm
twmstop = 0.0*Twm
gravwmtime = fd.Constant(0.0)
def gravwavemakertime(t,sigma,twmstop):
    if t<twmstop:
        return 0.0*np.sin(sigma*t)
    else:
        return 0.0
gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))
gravwm = Amp*fd.conditional(x[1]<L1,(x[1]-L1)/L1*gravwmtime,0.0)
mu = H0**2 # Needs checking; checked!
# 
deltacons = Zn-hn-Z0+h0
nwave = 0
if nwave==1:
    Forcingterm = fd.exp(-0.5*cc*(Zn-hn+Z0-h0)+cc*(Keel+tanalp*(x[1]-Ly)))*fd.conditional(fd.eq(deltacons,0.0),1.0,-ccinv*( 2.0*fd.sinh(-0.5*cc*deltacons) )/deltacons)
else:
    Forcingterm = 0.0
phi_expr = ( -vvmp0*(hn-h0)/dt-0.5*mu*fd.inner(fd.grad(vvmp0),fd.grad(hn-h0))/dt \
           +fd.inner( fd.grad(vvmp0),(1/6)*(2*h0*fd.grad(phi0)+2*hn*fd.grad(phin)+h0*fd.grad(phin)+hn*fd.grad(phi0)) ) \
           +mu*fd.inner(0.5*fd.grad(qn+q0),fd.grad(vvmp0)) )*fd.dx(degree=vpolyp) # nil/rest solution
h_expr   = (  vvmp1*(phin-phi0)/dt+0.5*mu*fd.inner(fd.grad(vvmp1),fd.grad(phin-phi0))/dt \
                +vvmp1*(1/6)*( fd.inner(fd.grad(phi0),fd.grad(phi0))+fd.inner(fd.grad(phin),fd.grad(phin))+fd.inner(fd.grad(phi0),fd.grad(phin))  ) \
                +vvmp1*(grav*(0.5*(hn+h0)-H0)+ccinv*Forcingterm + grav*gravwm )  )*fd.dx(degree=vpolyp)
# hnlmassterm = (1/6)*( 2*h0*fd.grad(phi0)+2*hn*fd.grad(hn)+h0*fd.grad(hn)+hn*fd.grad(h0) )
q_expr   = ( 0.5*fd.inner(fd.grad(phin)+fd.grad(phi0),fd.grad(vvmp2))-vvmp2*0.75*(qn+q0) )*fd.dx(degree=vpolyp)   # nil/rest solution
if nwave==1:
    Z_expr   = ( vvmp3*( Area*meff*(Wn-W0)/dt+Area*meff*grav-ccinv*Forcingterm ))*fd.dx(degree=vpolyp)
    W_expr   = ( vvmp4*Area*( (Zn-Z0)/dt-0.5*(Wn+W0) ))*fd.dx(degree=vpolyp)   # nil/rest solution
else:
    Z_expr   = ( vvmp3*( Zn ))*fd.dx(degree=vpolyp)
    W_expr   = ( vvmp4*( Wn ))*fd.dx(degree=vpolyp)   # nil/rest solution

F_exprnl = phi_expr+h_expr+q_expr+Z_expr+W_expr

solver_parameters4 = {
    'snes_type': 'newtonls',
    "snes_monitor": None,
    'snes': {
        'rtol': 1e-9,
        'atol': 1e-10,
        'max_it': 100,
        "converged_reason": None,
    },
    'mat_type': 'nest',
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_0_fields': '0,1,2',  # Continuous fields h, phi, q
    'pc_fieldsplit_1_fields': '3,4',    # R variables Z, W
    'pc_fieldsplit_0': {
        'type': 'schur',
        'schur_factorization_type': 'full',
        'schur_precondition': 'selfp',
    },
    'pc_fieldsplit_1': {
        'mat_type': 'nest',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    },
}

# Hydrostatic solver for initial condition:
# wavebuoy_hydrotstaticnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_hyexprnl, result_mixedmp))
wavebuoy_hydrotstaticnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_hyexprnl, result_mixedmp), solver_parameters=solver_parameters4)
# Dynamic solver:
#wavebuoy_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_exprnl, result_mixedmp), solver_parameters=solver_parameters4)
#
#
wavebuoy_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_exprnl, result_mixedmp))

# lines_parameters = {'ksp_type': 'gmres', 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC', 'star_construct_dim': 2,'star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}

###### OUTPUT FILES and initial PLOTTING ##########
save_path =  "data/"
if not os.path.exists(save_path):
    os.makedirs(save_path) 
#outfile_phin = fd.File(save_path+"phin.pvd")
outfile_phin = VTKFile(os.path.join(save_path, "phin.pvd"))
#outfile_hn = fd.File(save_path+"hn.pvd") #outfile_qn = fd.File(save_path+"qn.pvd")
fileE = save_path+"potflow3dperenergy.txt"
# outputE = open(fileE,"w") # outputEtamax = open(filEtaMax,"w") # outfile_height.write(h_old, time=t)
# outfile_psi.write(psi_f, time=t) # outfile_varphi.write(varphi, time=t)

# Solve hydrostatic rest state:
cc_values = [100.0, 500.0, 1000.0, 2000.0, 4000.0] # , 4000.0, 5000.0, 7000.0] # , 7000.0, 10000.0]
# --- Continuation Loop Execution ---
print("Starting continuation loop...")
for cc_val in cc_values:
    # Update the Firedrake Constant with the new cc value
    cc.assign(cc_val)
    print(f"Solving for cc = {cc_val}...")
    solver_reason = wavebuoy_hydrotstaticnl.solve()
    phin, hn, qn, Zn, Wn = fd.split(result_mixedmp)
    print(f"Solver for cc={cc_val} converged reason: {solver_reason}")

    
# Plotting hydrostatic initial condition state
plt.figure(2)
t = 0.0
ii = 0
if nmpi==0:
    plt.ion() 
# fig, (ax1,ax2,ax3,ax4) = plt.subplots(2,2)
fig, axes = plt.subplots(2, 2)
# axes is a 2D array, so you can access each plot by its row and column index
ax1 = axes[0, 0] # Top-left plot
ax2 = axes[0, 1] # Top-right plot
ax3 = axes[1, 0] # Bottom-left plot
ax4 = axes[1, 1] # Bottom-right plot
 
Z1.interpolate(Zn)
W1.interpolate(Wn)
h1.interpolate(hn)
phi1.interpolate(phin)
q1.interpolate(qn)
Z00 = np.array(Z1.vector())
W00 = np.array(W1.vector())
phi1vals = np.array([phi1.at(xslice,y) for y in yvals])
h1vals = np.array([h1.at(xslice,y) for y in yvals])
q1vals = np.array([q1.at(xslice,y) for y in yvals])
ax1.plot(yvals,h1vals,'-')
ax1.plot(yvals,0.0*h1vals,'-r')
ax2_twin = ax2.twinx()
ax2_twin.plot(yvals,phi1vals,'-')
ax3.plot(yvals,q1vals,'-')
ax4_twin = ax4.twinx()
ax4.plot(t, Z00, 'b.')
ax4.set_ylabel('Z(t)', color='b')
ax4.tick_params(axis='y', labelcolor='b')
ax4_twin.plot(t, W00, 'rx', label='W(t)')
ax4_twin.tick_params(axis='y', labelcolor='r')
ax4_twin.set_ylabel('Wt)', color='r')
# ax4.legend() # Legend to distinguish Z and W
# Plot buoy shape:
Lw = 0.1*Ly
yvLw = np.linspace(Ly-Lw+10**(-10), Ly-10**(-10), ny)
hbuoy0 = np.heaviside(yvLw-Ly+Lw,0.5)*(Z00-Keel-tanalp*(yvLw-Ly))
ax1.plot(yvLw,hbuoy0,'--r')
tsize2 = 12 # font size of image axes
size = 10   # font size of image axes
# ax1.set_xticklabels([]) # removed
ax1.set_xlim(xmin=0.8*Ly, xmax=Ly)
ax1.set_ylim(ymin=0, ymax=1.5*H0)
ax2.set_xticklabels([]) # removed
fig.suptitle(r'BLE-wave-buoy, energy-conserving AVF, Firedrake:',fontsize=tsize2)   
ax1.set_ylabel(r'$h(\frac {1}{2} L_x,y,t), h_b(Z(t),y) $ ',fontsize=size)
ax2_twin.set_ylabel(r'$\phi(\frac {1}{2} L_x,y,t)$ ',fontsize=size)
ax3.set_ylabel(r'$q(\frac {1}{2} L_x,y,t)$ ',fontsize=size)
ax3.set_xlabel(r'$y$ ',fontsize=size)
ax4.set_xlabel(r'$t$ ',fontsize=size)
# ax5.set_ylabel(r'$W(t)$ ',fontsize=size)

# Initial condition is hydrostatic rest state just calculated
phi0.assign(phi1)
h0.assign(h1)
q0.assign(q1)
W0.assign(W1)
Z0.assign(Z1)

E0 = fd.assemble( ( 0.5*hn*fd.inner(fd.grad(phin),fd.grad(phin))+mu*(fd.inner(fd.grad(phin),fd.grad(qn))-0.75*q1**2)+0.5*grav*fd.inner(hn-H0,hn-H0) \
                    +(nwave/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) + nwave*Area*meff*(0.5*Wn**2+grav*Zn) )*fd.dx(degree=vpolyp) )
print('IC E0:',E0,grav)

gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))

nplotyes =1
tic = tijd.time()
# Time loop starts: Needs to start with Zn,Wn,phim,hn, qn of hydrostatic solve
while t <= 1.0*(t_end + dt):
    #
    gravwmtime.assign(gravwavemakertime(t+0.5*dt,sigma,twmstop))
    wavebuoy_nl.solve()
    phin, hn, qn, Zn, Wn = fd.split(result_mixedmp)
    ii = ii+1
    t+= dt
    Z1.interpolate(Zn)
    W1.interpolate(Wn)
    h1.interpolate(hn)
    phi1.interpolate(phin)
    q1.interpolate(qn)

    # print('ii, t',ii,t)
    # Plotting
    #
    if (t in t_plot): # 
        # print('Plotting starts')
        # if nplotyes==1:
        #
        print('ii, t',ii,t)
        Z00 = np.array(Z1.vector())
        W00 = np.array(W1.vector())
        phi1vals = np.array([phi1.at(xslice,y) for y in yvals])
        h1vals = np.array([h1.at(xslice,y) for y in yvals])
        q1vals = np.array([q1.at(xslice,y) for y in yvals])
        ax1.plot(yvals,h1vals,'-')
        ax1.plot(yvals,0.0*h1vals,'-r')
        ax2_twin.plot(yvals,phi1vals,'-')
        ax3.plot(yvals,q1vals,'-')
        ax4.plot(t, Z00, 'b.', label='Z(t)')
        ax4_twin.plot(t, W00, 'rx', label='W(t)')
        # ax4.plot(t,Z00,'.')
        E1 = fd.assemble( ( 0.5*hn*fd.inner(fd.grad(phin),fd.grad(phin))+mu*(fd.inner(fd.grad(phin),fd.grad(qn))-0.75*qn**2)+0.5*grav*fd.inner(hn-H0,hn-H0) \
                            +(nwave/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) + nwave*Area*meff*(0.5*Wn**2+grav*Zn) )*fd.dx(degree=vpolyp) )
        print('E1, |E1-E0|/E0:',E1, np.abs(E1-E0)/E0)
        # ax5.plot(t,W00,'.')
        hbuoy0 = np.heaviside(yvLw-Ly+Lw,0.5)*(Z00-Keel-tanalp*(yvLw-Ly))
        ax1.plot(yvLw,hbuoy0,'--r') # Plot buoy shape
        if nmpi==0:
            plt.draw()
            plt.pause(0.01)
    
    # Copy new state to old state for next time step
    phi0.assign(phi1)
    h0.assign(h1)
    q0.assign(q1)
    W0.assign(W1)
    Z0.assign(Z1)
    
# End while time loop
toc = tijd.time() - tic
#
if nprintout==1:
    print('Elapsed time (min):', toc/60)
else:
    PETSc.Sys.Print('Elapsed time (min):', toc/60)
    
#outfile_phin.close()
#outfile_hn.close()
#outfile_qn.close()

plt.savefig("figs/wavebuoyavfc2025.png")

#
if nmpi==0:
    plt.show()


if nprintout==1:
    print('*************** PROGRAM ENDS ******************')
else:
    PETSc.Sys.Print('*************** PROGRAM ENDS ******************')
