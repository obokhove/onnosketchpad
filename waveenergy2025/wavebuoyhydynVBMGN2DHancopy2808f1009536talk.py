#
# 2D horizontal Variational Boussinesqg or Green-Naghdi water-wave equations in x-periodic channel based on implemenation with VP
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
nserial=1
if nmpi==0:
    import platform
    if platform.system() == "Darwin":   # macOS
        matplotlib.use("MacOSX")
        # matplotlib.use("Agg")
    else:  # Linux / Docker
        matplotlib.use("Agg")  # non-interactive, safe everywhere matplotlib.use('MacOSX') use pip3 install Pillow for animation
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import os.path
from parafull import a_rad, Li, gam, alp, Hm, L, Rc, Ri, Rl
from firedrake import *
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from finat.point_set import PointSet, GaussLegendrePointSet, GaussLobattoLegendrePointSet
from finat.quadrature import QuadratureRule, TensorProductQuadratureRule
nbah = 1
if nbah==1:
    from meshingbah import get_mesh
else:
    from meshing import get_mesh
from petsc4py import PETSc
import matplotlib.animation as animation
from PIL import Image # THIS LINE WAS ADDED
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
    # ggrav = 1.0 # grav
    ggrav = fd.Constant(1.0)
    thetac = np.pi*68.26/180
    Lc = Ly-0.5*Lx*np.tan(thetac)
    rho0 = 997
    Mm = 0.283*2
    alp3 = 0.5389316
    tan_th = (Ly-Lc)/(0.5*Lx)
    Keel = 0.04
    tanalp = np.tan(alp3)
    meff = Mm/rho0
    Zbars = H0+Keel-( 3*Mm*np.tan(thetac)*np.tan(alp3)**3/rho0 )**(1/3)
    print('a_rad, Li, gam, alp, Hm, L, Rc, Ri, Rl: ',a_rad, Li, gam, alp, Hm, L, Rc, Ri, Rl)
    
    
    
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
# ERROR 07-09-2025:
Area = 1/Area #  circa factor 25
print('Lx etc. ... tanalp',Lx,Ly,Lr,Area,tanalp)

if nserial==1:
    mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)[0]
    mesh = mesh2d
    # Save using checkpoint
    with fd.CheckpointFile("custom_mesh.h5", "w") as chk:
        chk.save_mesh(mesh)
else:
    # Load using checkpoint
    with fd.CheckpointFile("custom_mesh.h5", "r") as chk:
        mesh2d = chk.load_mesh()
    # mesh2d = fd.Mesh("custom_mesh.h5")
    mesh = mesh2d
# x, y, z = fd.SpatialCoordinate(mesh)

print('Hallo0')

nmeshplot=0
if nmeshplot==1:
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
    nnopen = 1
    if nnopen==1:
        from PIL import Image
        Image.open('mesh2d.png').rotate(-90, expand=True).save('mesh2d.png')

        
x1, y = fd.SpatialCoordinate(mesh2d)
x = mesh2d.coordinates

print('Hallo')

top_id = 'top'
seps = 0.0 #  -10**(-10)
yvals = np.linspace(0.0+seps, Ly-seps, nCG*ny)
xslice = 0.5*Lx
# 

t = 0
fac = 1.0 # Used to split h=H0+eta in such in way that we can switch to solving h (fac=0.0) or eta (fac=1.0)
if nic=="rest":    
    ttilde = 50 # time units 
    t_end = 4.0 #  0.5 # 4.0 0.8*1.5
    # t_end = 1.2
    Nt = 40
    dt = t_end/Nt #
    dtt = 1.2/Nt
    nnCG = (nCG+1)**2/4 # i.e (nCG+1)**2/4
    dt3 = np.minimum(Lx/nx,ny/Ly)/np.sqrt(grav*H0)/nnCG 
    CFL1 = 0.25*0.125
    CFL = 0.7*0.125 # was 0.125
    dt = CFL*dt3
    print('time steps',dt,CFL1*dtt,CFL1*dt3,dt3)
    nplot = 80  # 10
    tijde = []
    while (t <= t_end+1*dt):
        tijde.append(t)
        t+= dt
    nt = int(len(tijde)/nplot)
    t_plot = tijde[0::nt]
    plt.pause(0.002)

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
I0 = fd.Function(V_R, name="I0") # W at level n

phi1 = fd.Function(V_W, name="phi1") # velocity potential at level n+1 at free surface
h1 = fd.Function(V_W, name="h1") # water depth at level n+1
q1 = fd.Function(V_W, name="q1") # q at level n+1
Z1 = fd.Function(V_R, name="Z1") # Z at level n+1
W1 = fd.Function(V_R, name="W1") # W at level n+1
I1 = fd.Function(V_R, name="I1") # W at level n+1
lam1 = fd.Function(V_W, name="lam1") # lambda via penalty approach at level n+1

if nvpcase=="AVF":
    # Variables at midpoint for modified midpoint waves
    mixed_Vmp = V_W * V_W * V_W * V_R * V_R * V_R
    result_mixedmp = fd.Function(mixed_Vmp)
    vvmp = fd.TestFunction(mixed_Vmp)
    vvmp0, vvmp1, vvmp2, vvmp3, vvmp4, vvmp5 = fd.split(vvmp) # These represent "blocks".
    phin, hn, qn, Zn, Wn, In= fd.split(result_mixedmp) # 

# Initialise variables; projections on main variables at initial time
vpolyp = 15
cc = fd.Constant(100)
ccinv = cc**(-1)
# VP or weak forms hydrostatic rest case
phi_hyexpr = ( vvmp0*( phin ))*fd.dx(degree=vpolyp) # nil/rest solution
h_hyexpr   = ( vvmp1*( grav*(hn-H0)+ggrav*ccinv*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) ))*fd.dx(degree=vpolyp)
q_hyexpr   = ( vvmp2*( qn ))*fd.dx(degree=vpolyp)   # nil/rest solution
Z_hyexpr   = ( vvmp3*( Area*Mm*grav-rho0*ggrav*ccinv*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) ))*fd.dx(degree=vpolyp)
W_hyexpr   = ( vvmp4*( Wn ))*fd.dx(degree=vpolyp)   # nil/rest solution
I_hyexpr   = ( vvmp5*( In ))*fd.dx(degree=vpolyp)   # nil/rest solution
F_hyexprnl = phi_hyexpr+h_hyexpr+q_hyexpr+Z_hyexpr+W_hyexpr+I_hyexpr

# VP or weak forms dynamic case: first test is rest state stays rest state; next test add forcing
Amp = 0.025 #  0.025
L1 = 0.2
Twm = 1.0
sigma = 2.0*np.pi/Twm
twmstop = 2.0*Twm
twmstop= 2.0
gravwmtime = fd.Constant(0.0)
def gravwavemakertime(t,sigma,twmstop):
    if t<twmstop:
        return np.sin(sigma*t)
    else:
        return 0.0
gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))
gravwm = Amp*fd.conditional(x[1]<L1,(x[1]-L1)/L1*gravwmtime,0.0)
mu = H0**2 # Needs checking; checked!
# 
deltacons = Zn-hn-Z0+h0
nwave = 1
if nwave==1:
    Forcingterm = ggrav*fd.exp(-0.5*cc*(Zn-hn+Z0-h0)+cc*(Keel+tanalp*(x[1]-Ly)))*fd.conditional(fd.eq(deltacons,0.0),1.0,-ccinv*( 2.0*fd.sinh(-0.5*cc*deltacons) )/deltacons)
else:
    Forcingterm = 0.0

# Shit approach
nsa = 3
if nsa==0:
    delHamdelphi = ( fd.inner( fd.grad(vvmp0),(1/6)*(2*h0*fd.grad(phi0)+2*hn*fd.grad(phin)+h0*fd.grad(phin)+hn*fd.grad(phi0)) )  )*fd.dx(degree=vpolyp)
    delHamdelh = ( vvmp1*(1/6)*( fd.inner(fd.grad(phi0),fd.grad(phi0))+fd.inner(fd.grad(phin),fd.grad(phin))+fd.inner(fd.grad(phi0),fd.grad(phin))  ) \
               +vvmp1*(grav*(0.5*(hn+h0)-H0))  )*fd.dx(degree=vpolyp)
    print(' BEWARE: Shallow-water test. Do not use this option!')
else:
    ah01=1.0
    aq01=1.0
    aq02=1.0
    betaa=1.0
    n_quad_points = 4
    s_points, s_weights = np.polynomial.legendre.leggauss(n_quad_points)
    sufl = fd.Constant(0.0)
    si = (s_points+1)/2
    wi = s_weights/2
    hs = h0 + sufl*(hn-h0) 
    phis = phi0 + sufl*(phin-phi0)
    qs = q0 + sufl*(qn-q0)
    ubars = fd.grad(phis)+ah01*hs*qs*fd.grad(hs)+aq01*(1/3)*hs**2*fd.grad(qs)
    H_s = ( 0.5*hs*fd.inner(fd.grad(phis)+ah01*hs*qs*fd.grad(hs)+aq01*(1/3)*hs**2*fd.grad(qs),\
                            fd.grad(phis)+ah01*hs*qs*fd.grad(hs)+aq01*(1/3)*hs**2*fd.grad(qs))+\
            aq02*(1/6)*hs**3*qs**2+\
            0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )
    #  s-Integrate the Hamiltonian density: loop evaluates Hamiltonian at each Gaussian point and sums weighted results
    H_integrated = (fd.Constant(0.0)*hs)*fd.dx(degree=vpolyp)
    H_s_evaluated = []
    for ii in range(n_quad_points):
        # Replace symbolic 's_ufl' with numeric 'si' value for iteration: old: H_s_evaluated = fd.replace(H_s, {sufl: si[ii]}) # OLD
        H_s_evaluated.append(fd.replace(H_s, {sufl: si[ii]}))
        H_integrated += wi[ii]*H_s_evaluated[ii]*fd.dx(degree=vpolyp)  # Accumulate weighted Hamiltonian value
        # s-integrated Hamiltonian is a UFL Form object:
    Hamiltonian = H_integrated
    delHamdelphi = fd.derivative(Hamiltonian,phi0,du=vvmp0)+fd.derivative(Hamiltonian,phin,du=vvmp0)
    delHamdelh = fd.derivative(Hamiltonian,h0,du=vvmp1)+fd.derivative(Hamiltonian,hn,du=vvmp1)
    if aq01>0:
        if aq02>0:
            delHamdelpsi = fd.derivative(Hamiltonian,q0,du=vvmp2)+fd.derivative(Hamiltonian,qn,du=vvmp2)

if nsa == 1:    
    phi_expr = ( vvmp0*(hn-h0)/dt \
                 -fd.inner( fd.grad(vvmp0),(1/6)*(2*h0*fd.grad(phi0)+2*hn*fd.grad(phin)+h0*fd.grad(phin)+hn*fd.grad(phi0)) )  )*fd.dx(degree=vpolyp) # nil/rest solution
    h_expr   = (  vvmp1*(phin-phi0)/dt +\
                  vvmp1*(1/6)*( fd.inner(fd.grad(phi0),fd.grad(phi0))+fd.inner(fd.grad(phin),fd.grad(phin))+fd.inner(fd.grad(phi0),fd.grad(phin))  )+ \
                  vvmp1*(grav*(0.5*(hn+h0)-H0)+ ccinv*Forcingterm + grav*gravwm )  )*fd.dx(degree=vpolyp)
    print(' BEWARE: Note shallow-water limit test. Do not use this option!')
else:
    print('Hallo nsa',nsa)
    phi_expr = ( vvmp0*(hn-h0)/dt )*fd.dx(degree=vpolyp) - delHamdelphi
    h_expr   = (  vvmp1*(phin-phi0)/dt+\
                  vvmp1*( ccinv*Forcingterm + grav*gravwm ) )*fd.dx(degree=vpolyp) + delHamdelh
     
# hnlmassterm = (1/6)*( 2*h0*fd.grad(phi0)+2*hn*fd.grad(hn)+h0*fd.grad(hn)+hn*fd.grad(h0) )
if nsa==3:
    if aq01>0:
        q_expr = delHamdelpsi
    else:
        q_expr   = ( vvmp2*qn )*fd.dx(degree=vpolyp)    # nil/rest solution
        print(' BEWARE: Note no psi-dynamics test. Do not use this option!')
else:
    q_expr   = ( vvmp2*qn )*fd.dx(degree=vpolyp)   # nil/rest solution
    print(' BEWARE:  Note no psi-dynamics test. Do not use this option!')

Zbar = fd.Constant(0.07677428)  # Should be based on hydrostatic case result
Z12 = 0.5*(Zn+Z0)
GZZ12 = 1.0/(a_rad**2 + (Zbar + alp*Hm - (Z12) -L/2)**2)**(3/2) - 1.0/(a_rad**2 + (Zbar + alp*Hm - (Z12) + L/2)**2)**(3/2)
print('Zbar after cst',Zbar)
#  gam = 0.0
if nwave==1:
    # Z_expr   = ( vvmp3*( Area*meff*(Wn-W0)/dt+Area*meff*grav-ccinv*Forcingterm ))*fd.dx(degree=vpolyp)
    Z_expr   = ( vvmp3*( Area*meff*(Wn-W0)/dt+Area*meff*grav-ccinv*Forcingterm+Area*gam*GZZ12*0.5*(In+I0)/rho0 ))*fd.dx(degree=vpolyp)
    W_expr   = ( vvmp4*Area*( (Zn-Z0)/dt-0.5*(Wn+W0) ))*fd.dx(degree=vpolyp)   # nil/rest solution
    # I_expr   = ( vvmp5*Area*( Li*(In-I0)/dt-0.5*(Wn+W0) ))*fd.dx(degree=vpolyp)   # nil/rest solution
    I_expr   = ( vvmp5*Area*( Li*(In-I0)/dt-gam*GZZ12*0.5*(Wn+W0)+(Rc+Ri+Rl)*0.5*(In+I0) ))*fd.dx(degree=vpolyp)   # nil/rest solution TO DO define Li, gam, GZZ12, a_rad, Zbar, alp, Hm, Z12
else:
    Z_expr   = ( vvmp3*( Zn ))*fd.dx(degree=vpolyp)
    W_expr   = ( vvmp4*( Wn ))*fd.dx(degree=vpolyp)   # nil/rest solution
    I_expr   = ( vvmp5*( In ))*fd.dx(degree=vpolyp)

F_exprnl = phi_expr+h_expr+q_expr+Z_expr+W_expr+I_expr

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
def monitor(snes, its, fnorm):
    PETSc.Sys.Print(f"  SNES it {its}, residual norm {fnorm:.4e}")
# Access PETSc SNES object
snes = wavebuoy_nl.snes
# Set tolerances directly on the SNES object
snes.setTolerances(rtol=1e-16, atol=1e-16)
# snes.setType("vinewtonrsls")

# lines_parameters = {'ksp_type': 'gmres', 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC', 'star_construct_dim': 2,'star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}

###### OUTPUT FILES and initial PLOTTING ########## /Users/onnobokhove/amtob/werk/vuurdraak2021/wavenergy2025
save_path =  "data9/"
if not os.path.exists(save_path):
    os.makedirs(save_path) 
outfile_phin = VTKFile(os.path.join(save_path, "phin.pvd"))
outfile_hn = VTKFile(os.path.join(save_path, "hn.pvd"))
outfile_psin = VTKFile(os.path.join(save_path, "psin.pvd"))
outfile_lambd = VTKFile(os.path.join(save_path, "lambd.pvd"))
fileE = os.path.join(save_path, 'VBMZWIenergy.csv')
filEdata = np.empty((0,7))
format_E = '%10.4f'*4+'%15.12f'*3

# Solve hydrostatic rest state:
print('before ggrav type:', type(ggrav))
print('before ggrav value:', float(ggrav))
cc_values = [10, 100.0, 500.0, 1000.0, 2000.0, 4000.0] # , 4000.0, 5000.0, 7000.0] # , 7000.0, 10000.0]
# --- Continuation Loop Execution ---
print("Starting continuation loop...")
for cc_val in cc_values:
    # Update the Firedrake Constant with the new cc value
    cc.assign(cc_val)
    print(f"Solving for cc = {cc_val}...")
    solver_reason = wavebuoy_hydrotstaticnl.solve()
    phin, hn, qn, Zn, Wn, In = fd.split(result_mixedmp)
    print(f"Solver for cc={cc_val} converged reason: {solver_reason}")

ggrav.assign(grav)
solver_reason = wavebuoy_hydrotstaticnl.solve()
phin, hn, qn, Zn, Wn, In = fd.split(result_mixedmp)
print(f"Solver for cc={cc_val} converged reason: {solver_reason}")
print('ggrav type:', type(ggrav))
print('ggrav value:', float(ggrav))
    
if nserial==1:
    # --- User choice for plotting mode ---
    print('Choose plotting mode:')
    print('1. Static (shows final plot only)')
    print('2. Animated (shows plots in real-time)')
    plot_mode_choice = input('Enter 1 or 2: ')
    animate_mode = (plot_mode_choice == '2')
    animation_data = []  # This will store the actual data, not canvas regions
else: # MPI
    animate_mode = False
    
Lw = 0.1*Ly
yvLw = np.linspace(Ly-Lw+10**(-10), Ly-10**(-10), ny)

def update_plotstatic(fig, axes, fig3, ax1_fig3, t, Z00, W00, h1vals, phi1vals, q1vals, yvals, yvLw, Lw, Ly, Z1, Keel, tanalp):
    print('Hallo static')
    #ax1 = axes[0, 0]
    #ax2 = axes[0, 1]
    #ax3 = axes[1, 0]
    #ax4 = axes[1, 1]
    #ax2_twin = ax2.twinx()
    #ax4_twin = ax4.twinx()
    # Plotting logic
    # plt.figure(2)
    ax1.plot(yvals, h1vals, '-', label='$h$')
    #ax1.plot(yvals, 0.0 * h1vals, '-r', label='Water level')
    ax2_twin.plot(yvals, phi1vals, '-', color='tab:blue', label=r'$\phi$')
    ax3.plot(yvals, q1vals, '-', label=r'$\psi$')
    ax4.plot(t, Z00, '.b', label='$Z(t)$')
    ax4_twin.plot(t, W00, 'xr', label='$W(t)$')
    hbuoy0 = np.heaviside(yvLw - Ly + Lw, 0.5) * (Z00 - Keel - tanalp * (yvLw - Ly))
    ax1.plot(yvLw, hbuoy0, '--r', label='buoy')
    # Update labels and limits
    tsize2 = 12
    size = 10
    fig.suptitle(r'BLE-wave-buoy, energy-conserving AVF, Firedrake:', fontsize=tsize2)
    ax1.set_ylabel(r'$h(\frac{1}{2}L_x,y,t)$, $h_b(Z(t),y)$', fontsize=size)
    ax2_twin.set_ylabel(r'$\phi(\frac{1}{2}L_x,y,t)$', fontsize=size)
    ax3.set_ylabel(r'$\psi(\frac{1}{2}L_x,y,t)$', fontsize=size)
    ax3.set_xlabel(r'$y$', fontsize=size)
    ax4.set_xlabel(r'$t$', fontsize=size)
    ax4.set_ylabel(r'$Z(t)$', color='b', fontsize=size)
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.set_ylabel(r'$W(t)$', color='r', fontsize=size)
    ax4_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_xlim(xmin=0.8 * Ly, xmax=Ly)
    ax1.set_ylim(ymin=0, ymax=1.5 * H0)
    #ax1.legend(loc='upper right', fontsize=size) # Add legends
    #ax4.legend(loc='lower left', fontsize=size)
    #ax4_twin.legend(loc='upper left', fontsize=size)
    # plt.figure(3) # Plot for the new figure (fig3)
    ax1_fig3.plot(yvals, h1vals, '-', label='$h$')
    ax1_fig3.plot(yvLw, hbuoy0, '--r', label='buoy')
    ax1_fig3.set_xlabel(r'$y$', fontsize=size)
    ax1_fig3.set_ylabel(r'$h(\frac{1}{2}L_x,y,t)$, $h_b(Z(t),y)$', fontsize=size)
    ax1_fig3.set_title(r'$h$ and buoy shape across centre-line tank', fontsize=tsize2)
    ax1_fig3.set_xlim(xmin=0.0, xmax=Ly)
    ax1_fig3.set_ylim(ymin=0, ymax=1.5 * H0)
    #ax1_fig3.legend(loc='upper right', fontsize=size)

def update_plots(fig, axes, fig3, ax1_fig3, t, Z00, W00, h1vals, phi1vals, q1vals, yvals, yvLw, Lw, Ly, Z1, Keel, tanalp, save_for_animation=False):
    #ax1 = axes[0, 0]
    #ax2 = axes[0, 1]
    #ax3 = axes[1, 0]
    #ax4 = axes[1, 1]
    print('Hallo nonstatic')
    if save_for_animation: # Save data for animation if requested
        hbuoy0 = np.heaviside(yvLw - Ly + Lw, 0.5) * (Z00 - Keel - tanalp * (yvLw - Ly))
        frame_data = {
            't': t,
            'h1vals': h1vals.copy(),
            'phi1vals': phi1vals.copy(), 
            'q1vals': q1vals.copy(),
            'Z00': Z00.copy(),
            'W00': W00.copy(),
            'hbuoy0': hbuoy0.copy(),
            'yvals': yvals.copy(),
            'yvLw': yvLw.copy()
        }
        animation_data.append(frame_data)
        return  # Don't plot during data collection phase
    if animate_mode: # Clear existing plots if in animated mode to prevent overlap
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ax1_fig3.cla()
    ax2_twin = ax2.twinx()
    ax4_twin = ax4.twinx()
    # Plotting logic
    # plt.figure(2)
    ax1.plot(yvals, h1vals, '-', label='$h$')
    #ax1.plot(yvals, 0.0 * h1vals, '-r', label='Water level')
    ax2_twin.plot(yvals, phi1vals, '-', color='tab:blue', label=r'$\phi$')
    ax3.plot(yvals, q1vals, '-', label=r'$\psi$')
    ax4.plot(t, Z00, '.b', label='$Z(t)$')
    ax4_twin.plot(t, W00, 'xr', label='$W(t)$')
    hbuoy0 = np.heaviside(yvLw - Ly + Lw, 0.5) * (Z00 - Keel - tanalp * (yvLw - Ly))
    ax1.plot(yvLw, hbuoy0, '--r', label='buoy')
    # Update labels and limits
    tsize2 = 12
    size = 10
    fig.suptitle(r'BLE-wave-buoy, energy-conserving AVF, Firedrake:', fontsize=tsize2)
    ax1.set_ylabel(r'$h(\frac{1}{2}L_x,y,t)$, $h_b(Z(t),y)$', fontsize=size)
    ax2_twin.set_ylabel(r'$\phi(\frac{1}{2}L_x,y,t)$', fontsize=size)
    ax3.set_ylabel(r'$\psi(\frac{1}{2}L_x,y,t)$', fontsize=size)
    ax3.set_xlabel(r'$y$', fontsize=size)
    ax4.set_xlabel(r'$t$', fontsize=size)
    ax4.set_ylabel(r'$Z(t)$', color='b', fontsize=size)
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.set_ylabel(r'$W(t)$', color='r', fontsize=size)
    ax4_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_xlim(xmin=0.8 * Ly, xmax=Ly)
    ax1.set_ylim(ymin=0, ymax=1.5 * H0)
    # ax1.legend(loc='upper right', fontsize=size) # Add legends
    # ax4.legend(loc='lower left', fontsize=size)
    ax4_twin.legend(loc='upper left', fontsize=size)
    # plt.figure(3) # Plot for the new figure (fig3)
    ax1_fig3.plot(yvals, h1vals, '-', label='$h$')
    ax1_fig3.plot(yvLw, hbuoy0, '--r', label='buoy')
    ax1_fig3.set_xlabel(r'$y$', fontsize=size)
    ax1_fig3.set_ylabel(r'$h(\frac{1}{2}L_x,y,t)$, $h_b(Z(t),y)$', fontsize=size)
    ax1_fig3.set_title(r'$h$ and buoy shape across centre-line tank', fontsize=tsize2)
    ax1_fig3.set_xlim(xmin=0.0, xmax=Ly)
    ax1_fig3.set_ylim(ymin=0, ymax=1.5 * H0)
    # ax1_fig3.legend(loc='upper right', fontsize=size)
    if animate_mode and nmpi==0: # Drawing and pausing for real-time animation
        plt.draw()
        plt.pause(0.00001)

# Plotting hydrostatic initial condition state plt.figure(2) plt.figure(2)
t = 0.0
ii = 0
if nmpi==0:
    plt.ion() 
# fig, (ax1,ax2,ax3,ax4) = plt.subplots(2,2)
fig, axes = plt.subplots(2, 2)
ax1 = axes[0, 0] # Top-left plot
ax2 = axes[0, 1] # Top-right plot
ax3 = axes[1, 0] # Bottom-left plot
ax4 = axes[1, 1] # Bottom-right plot
ax2_twin = ax2.twinx()
ax4_twin = ax4.twinx()
# --- New figure for ax1 only, full-width ---
# plt.figure(3)
fig3, ax1_fig3 = plt.subplots(figsize=(10, 5))

Z1.interpolate(Zn)
W1.interpolate(Wn)
h1.interpolate(hn)
phi1.interpolate(phin)
q1.interpolate(qn)
I1.interpolate(In)
Z00 = np.array(Z1.vector())
W00 = np.array(W1.vector())
I00 = np.array(I1.vector())
phi1vals = np.array([phi1.at(xslice,y) for y in yvals])
h1vals = np.array([h1.at(xslice,y) for y in yvals])
q1vals = np.array([q1.at(xslice,y) for y in yvals])
print('Z00, W00, I00', Z00, W00, I00)

Zn_R_component = Z1.sub(0)  # Firedrake Function corresponding to R
zbar_val = float(Zn_R_component.dat.data_ro[0])  # R-component is a single number
Zbar = fd.Constant(zbar_val)
print('Zbar after Zn',Zbar)

# Make plot to check: or list
Nzpl = 100
zplmax = 0.2
zpl = np.linspace(0.0,zplmax,Nzpl)
Zbarv = float(Zbar)
GZZ12plt = 1.0/(a_rad**2 + (Zbarv + alp*Hm - (zpl) -L/2)**2)**(3/2) - 1.0/(a_rad**2 + (Zbarv + alp*Hm - (zpl) + L/2)**2)**(3/2)
print('Check: zpl, G(zpl):',zpl, GZZ12plt)
plt.figure(20)
plt.plot(zpl,GZZ12plt,'-')
plt.xlabel("Z")
plt.ylabel("G(Z)")
plt.legend()
plt.savefig('GZ.png')

# Initial plot update
if animate_mode and nmpi==0:
    print('Hallo nonstatic')
    update_plots(fig, axes, fig3, ax1_fig3, t, Z00, W00, h1vals, phi1vals, q1vals, yvals, yvLw, Lw, Ly, Z1, Keel, tanalp,save_for_animation=True)
else:
    print('Hallo static')
    if nmpi==0:
        update_plotstatic(fig, axes, fig3, ax1_fig3, t, Z00, W00, h1vals, phi1vals, q1vals, yvals, yvLw, Lw, Ly, Z1, Keel, tanalp)
    
# Initial condition is hydrostatic rest state just calculated
phi0.assign(phi1)
h0.assign(h1)
q0.assign(q1)
W0.assign(W1)
Z0.assign(Z1)
I0.assign(I1)
lam1.interpolate( -(nwave/cc)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) )
outfile_phin.write(phi1, time=t)
outfile_psin.write(q1, time=t)
outfile_hn.write(h1, time=t)
outfile_lambd.write(lam1, time=t)

E0 = fd.assemble( ( 0.5*hn*fd.inner(fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn),fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn))+0.5*grav*fd.inner(hn-H0,hn-H0) \
                    +aq02*(1/6)*hn**3*qn**2+(betaa/90)*hn**5*fd.inner(fd.grad(qn),fd.grad(qn))+(nwave*ggrav/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly)))\
                    + nwave*Area*meff*(0.5*Wn**2+grav*Zn) + Area*0.5*Li*In**2/rho0  )*fd.dx(degree=vpolyp) )
gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))
tic = tijd.time()
nstop = 0
# Time loop starts: Needs to and does start with Zn,Wn,phim,hn,qn of hydrostatic solve as input
Gfunc = fd.Function(V_W, name="GZZ12")
while t <= 1.0*(t_end + dt):
    #
    gravwmtime.assign(gravwavemakertime(t+0.5*dt,sigma,twmstop))
    # print('TIME t and ii',t,ii)
    wavebuoy_nl.solve()
    phin, hn, qn, Zn, Wn, In = fd.split(result_mixedmp)
    ii = ii+1
    t+= dt
    Z1.interpolate(Zn)
    W1.interpolate(Wn)
    h1.interpolate(hn)
    phi1.interpolate(phin)
    q1.interpolate(qn)
    I1.interpolate(In)
    
    #
    if (t in t_plot): # Plotting # print('Plotting starts')
        #fd.project(GZZ12, V, function=Gfunc)
        #print("timestep",t, "GZZ12 min,max =", float(Gfunc.dat.data_ro.min()), float(Gfunc.dat.data_ro.max()), "gam =", float(gam))
        
        fnorm = snes.getFunctionNorm()
        PETSc.Sys.Print(f"Step {ii}, final SNES residual norm {fnorm:.4e}")
        PETSc.Sys.Print(" Converged reason:", snes.getConvergedReason())    
        print('Plotting: ii, t',ii,t)
        Z00 = np.array(Z1.vector())
        W00 = np.array(W1.vector())
        I00 = np.array(I1.vector())
        phi1vals = np.array([phi1.at(xslice,y) for y in yvals])
        h1vals = np.array([h1.at(xslice,y) for y in yvals])
        q1vals = np.array([q1.at(xslice,y) for y in yvals])
        # SWE expression: E1 = fd.assemble( ( 0.5*hn*fd.inner(fd.grad(phin),fd.grad(phin))+0.5*grav*fd.inner(hn-H0,hn-H0) \
        #                    +(nwave/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) + nwave*Area*meff*(0.5*Wn**2+grav*Zn) )*fd.dx(degree=vpolyp) )
        E1 = fd.assemble( ( 0.5*hn*fd.inner(fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn),fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn))+0.5*grav*fd.inner(hn-H0,hn-H0) \
                            + aq02*(1/6)*hn**3*qn**2+(betaa/90)*hn**5*fd.inner(fd.grad(qn),fd.grad(qn))+(nwave*ggrav/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly)))\
                            + nwave*Area*meff*(0.5*Wn**2+grav*Zn) + Area*0.5*Li*In**2/rho0 )*fd.dx(degree=vpolyp) )
        print('E0, E1, |E1-E0|/E0:',E0, E1, np.abs(E1-E0)/E0)
        fileEdata = np.array([[ t, Z00[0], W00[0], I00[0], E0, E1, np.abs(E1-E0)/E0 ]])
        if t>twmstop and nstop==0:
            E0 = E1
            nstop = 1
            print('time, Twm, E0, E1',t,twmstop,E0,E1)
        
        # Update plots - save data if we want animations
        if animate_mode and nmpi==0:
            update_plots(fig, axes, fig3, ax1_fig3, t, Z00, W00, h1vals, phi1vals, q1vals, yvals, yvLw, Lw, Ly, Z1, Keel, tanalp, save_for_animation=True)
        else:
            update_plotstatic(fig, axes, fig3, ax1_fig3, t, Z00, W00, h1vals, phi1vals, q1vals, yvals, yvLw, Lw, Ly, Z1, Keel, tanalp)
            if nmpi==0:  # Only show live plots in static mode
                plt.draw()
                plt.pause(0.00001) 
        #
        lam1.interpolate( -(nwave/cc)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) )
        outfile_phin.write(phi1, time=t)
        outfile_psin.write(q1, time=t)
        outfile_hn.write(h1, time=t)
        outfile_lambd.write(lam1, time=t)
        with open(fileE,'a') as pfileE:
            np.savetxt(pfileE, fileEdata, fmt=format_E)

    # Copy new state to old state for next time step
    phi0.assign(phi1)
    h0.assign(h1)
    q0.assign(q1)
    W0.assign(W1)
    Z0.assign(Z1)
    I0.assign(I1)
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
# --- Final save and show ---


# --- CORRECTED Animation Creation ---
def create_animations():
    """Create animations from collected data"""
    if not animation_data:
        print("No animation data collected!")
        return
        
    print(f"Creating animations from {len(animation_data)} frames...")
    
    # Create new figures for animation
    fig_anim, axes_anim = plt.subplots(2, 2, figsize=(12, 8))
    fig3_anim, ax1_fig3_anim = plt.subplots(figsize=(10, 5))
    
    # Store time series data for ax4 plot
    all_times = [frame['t'] for frame in animation_data]
    all_Z = [frame['Z00'][0] for frame in animation_data]  # Z00 is array, take first element
    all_W = [frame['W00'][0] for frame in animation_data]  # W00 is array, take first element
    
    def animate_main_panel(frame_idx):
        """Animation function for main 2x2 panel"""
        if frame_idx >= len(animation_data):
            return []
            
        frame = animation_data[frame_idx]
        ax1, ax2, ax3, ax4 = axes_anim.flat
        ax2_twin = ax2.twinx()
        ax4_twin = ax4.twinx()
        
        # Clear axes
        ax1.cla()
        ax2.cla() 
        ax3.cla()
        ax4.cla()
        #ax2_twin.cla() # <<< --- LINE CHANGED --- >>>
        #ax4_twin.cla() # <<< --- LINE CHANGED --- >>>
        
        # Clear twin axes from the previous frame. This is the fix.
        # Check if ax2 has children and clear them. This avoids the "jumping and writing over".
        # It gets the twin axes via the main axis's children list.
        if hasattr(ax2, 'right_ax'): # <<< --- CHANGED LINE --- >>>
            ax2.right_ax.cla() # <<< --- CHANGED LINE --- >>>
            
        if hasattr(ax4, 'right_ax'): # <<< --- CHANGED LINE --- >>>
            ax4.right_ax.cla() # <<< --- CHANGED LINE --- >>>
            
        
        # Create twin axes for ax2 and ax4
        #ax2_twin = ax2.twinx()
        #ax4_twin = ax4.twinx()
        
        # Plot data
        ax1.plot(frame['yvals'], frame['h1vals'], '-', label='$h$')
        ax1.plot(frame['yvals'], 0.0 * frame['h1vals'], '-r', label='Water level')
        ax1.plot(frame['yvLw'], frame['hbuoy0'], '--r', label='Buoy')
        
        ax2_twin.plot(frame['yvals'], frame['phi1vals'], '-', color='tab:blue', label=r'$\phi$')
        ax3.plot(frame['yvals'], frame['q1vals'], '-', label=r'$\psi$')
        
        # Plot time series up to current frame
        current_times = all_times[:frame_idx+1]
        current_Z = all_Z[:frame_idx+1] 
        current_W = all_W[:frame_idx+1]
        ax4.plot(current_times, current_Z, '.b', label='$Z(t)$')
        ax4_twin.plot(current_times, current_W, 'xr', label='$W(t)$')
        
        # Set labels and limits
        size = 10
        tsize2 = 12
        fig_anim.suptitle(f'BLE-wave-buoy, t={frame["t"]:.3f}', fontsize=tsize2)
        ax1.set_ylabel(r'$h(\frac{1}{2}L_x,y,t)$', fontsize=size)
        ax2_twin.set_ylabel(r'$\phi(\frac{1}{2}L_x,y,t)$', fontsize=size)
        ax3.set_ylabel(r'$q(\frac{1}{2}L_x,y,t)$', fontsize=size)
        ax3.set_xlabel(r'$y$', fontsize=size)
        ax4.set_xlabel(r'$t$', fontsize=size)
        ax4.set_ylabel(r'$Z(t)$', color='b', fontsize=size)
        ax4_twin.set_ylabel(r'$W(t)$', color='r', fontsize=size)
        
        ax1.set_xlim(xmin=0.8 * Ly, xmax=Ly)
        ax1.set_ylim(ymin=0, ymax=1.5 * H0)
        ax4.set_xlim(0, max(all_times))
        
        # Add legends
        ax1.legend(loc='upper right', fontsize=size)
        ax4.legend(loc='lower left', fontsize=size) 
        ax4_twin.legend(loc='upper left', fontsize=size)
        
        return []
    
    def animate_h_only(frame_idx):
        """Animation function for h-only plot"""
        if frame_idx >= len(animation_data):
            return []
            
        frame = animation_data[frame_idx]
        ax1_fig3_anim.cla()
        
        ax1_fig3_anim.plot(frame['yvals'], frame['h1vals'], '-', label='$h$')
        ax1_fig3_anim.plot(frame['yvLw'], frame['hbuoy0'], '--r', label='buoy')
        
        size = 10
        tsize2 = 12
        ax1_fig3_anim.set_xlabel(r'$y$', fontsize=size)
        ax1_fig3_anim.set_ylabel(r'$h(\frac{1}{2}L_x,y,t)$', fontsize=size)
        ax1_fig3_anim.set_title(f'$h$ and buoy, t={frame["t"]:.3f}', fontsize=tsize2)
        ax1_fig3_anim.set_xlim(xmin=0.0, xmax=Ly)
        ax1_fig3_anim.set_ylim(ymin=0, ymax=1.5 * H0)
        ax1_fig3_anim.legend(loc='upper right', fontsize=size)
        
        return []
    
    # Create animations
    anim1 = animation.FuncAnimation(fig_anim, animate_main_panel, frames=len(animation_data), 
                                   interval=200, blit=False, repeat=True)
    anim2 = animation.FuncAnimation(fig3_anim, animate_h_only, frames=len(animation_data),
                                   interval=200, blit=False, repeat=True)
    
    # Save animations
    os.makedirs("figs", exist_ok=True)
    anim1.save("figs/wavebuoyGNVBM_animated_panel.gif", writer='pillow', fps=5)
    anim2.save("figs/wavebuoyGNVBM_animated_h_only.gif", writer='pillow', fps=5)
    
    print("Animations saved successfully!")
    
    # Keep references to prevent garbage collection
    return anim1, anim2

# --- Final save and show ---
if animate_mode and animation_data:
    animations = create_animations()
else:
    # Static mode - create final plots
    update_plots(fig, axes, fig3, ax1_fig3, t, Z00, W00, h1vals, phi1vals, q1vals, 
                yvals, yvLw, Lw, Ly, Z1, Keel, tanalp)
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/wavebuoyGNVBM_static_panel.png")
    fig3.savefig("figs/wavebuoyGNVBM_static_h_only.png")
    if nmpi==0:
        plt.show()
        print('Hello shown.')


if nprintout==1:
    print('*************** PROGRAM ENDS ******************')
else:
    PETSc.Sys.Print('*************** PROGRAM ENDS ******************')
