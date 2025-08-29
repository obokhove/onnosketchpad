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
nmpi = 0
nprintout = 0 # print out on screen or in file; 1: on screen: 0: PETSC version
# if nmpi==0: matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import os
import os.path
from firedrake import *
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from finat.point_set import PointSet, GaussLegendrePointSet, GaussLobattoLegendrePointSet
from finat.quadrature import QuadratureRule, TensorProductQuadratureRule
if nmpi==0:
    from meshing import get_mesh
elif nmpi==1:
    from meshingmpi import get_mesh
import ufl
from petsc4py import PETSc

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
     
if nic=="rest": # rest test
    Lz = 20
    H0 = 0.15
    Hk = 0.04
    nx = 9 # 19 39
    ny = 51 # 201 401
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
    
#________________________ MESH (from Jonny Bolton, who translated my Matlab mesh)  _______________________#
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
#  quadrilateral=True, reorder=None,distribution_parameters=None,diagonal=None,comm=COMM_WORLD,name='mesh2d') # xx1, xx2, yy1, yy2
Lxx = Lx
Lyy = Ly
nxx = nx
nyy = ny
Lr = (Ly-Lc) # 0.2*Ly
Area = Lx*Ly-0.5*Lx*Lr
if nmpi==0:
    print('Lx etc. ... tanalp',Lx,Ly,Lr,Area,tanalp)
elif nmpi==1:
    PETSc.Sys.Print('Lx etc. ... tanalp',Lx,Ly,Lr,Area,tanalp)

if nmpi==0:
    mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)[0] #  mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)
elif nmpi==1: # Another failed GG
    mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)
elif nmpi==2: # INCORRECT one of google gemini
    comm = fd.COMM_WORLD # ADDED LINE: Get the global MPI communicator
    # ADDED LINE: Check if the current process is the root process (rank 0)
    if comm.rank == 0: # ADDED LINE: Only the root process runs the mesh generation
        mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)[0]
        if nmpi==0:
            print("Mesh generated successfully by rank 0.") # ADDED LINE: Print a confirmation message
        elif nmpi==1:
            PETSc.Sys.Print('Mesh generated successfully')
    else: # ADDED LINE: All other processes set mesh2d to None as a placeholder
        mesh2d = None
    mesh2d = comm.bcast(mesh2d, root=0) # ADDED LINE: Broadcast the mesh object from rank 0 to all other ranks

    
if nmpi==0:
    print("Mesh generated successfully by rank 0.") # ADDED LINE: Print a confirmation message
elif nmpi==1:
    PETSc.Sys.Print('Mesh generated successfully')
 
mesh = mesh2d # x, y, z = fd.SpatialCoordinate(mesh)
nplotmesh = 1
if nplotmesh==1:
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
    t_end = 4.0 # 0.8*1.5
    Nt = 20
    dt = t_end/Nt #
    dtt = 1.2/Nt
    dt3 = np.minimum(Lx/nx,ny/Ly)/np.sqrt(grav*H0)
    CFL1 = 0.25*0.125
    CFL = 0.5
    dt = CFL*dt3
    print('time steps',dt,CFL1*dtt,CFL1*dt3,dt3)
    nplot = 20
    nplotZW = 40
    tijde = []
    while (t <= t_end+1*dt):
        tijde.append(t)
        t+= dt
    nt = int(len(tijde)/nplot)
    ntZW = int(len(tijde)/nplotZW)
    t_plot = tijde[0::nt]
    t_plotZW = tijde[0::ntZW]
                     
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
Amp = 0.025 #  0.025
L1 = 0.2
Twm = 0.5 # 0.5
sigma = 2.0*np.pi/Twm
twmstop = 0.0*2.0*Twm # 0.0 no wavr forcing; >0 wave forcing
gravwmtime = fd.Constant(0.0)
def gravwavemakertime(t,sigma,twmstop):
    if t<twmstop:
        return 0.0*np.sin(sigma*t)
    else:
        return 0.0
gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))
gravwm = Amp*fd.conditional(x[1]<L1,(x[1]-L1)/L1*gravwmtime,0.0)
# 
deltacons = Zn-hn-Z0+h0
nwave = 0 # 0: no buoy dynamics; 1: buoy dynamics
betaa = 0.0 # switch turning on/off GN term
aq01 = 0.0 # switch turning on/off q/psi-term in umean
aq02 = 0.0 # switch turning on/off q/psi-term in kinetic energy no gradient
ah01 = 0.0 # switch turning on/off grad(h)-term in umean
if nwave==1:
    Forcingterm = fd.exp(-0.5*cc*(Zn-hn+Z0-h0)+cc*(Keel+tanalp*(x[1]-Ly)))*fd.conditional(fd.eq(deltacons,0.0),1.0,-ccinv*( 2.0*fd.sinh(-0.5*cc*deltacons) )/deltacons)
else:
    Forcingterm = 0.0
btopo = 0 # No topography

# Define AVF s-integration scheme using Gaussian quadrature: 4-pnt or 5-pnt rules sufficient for 7th order polynomial: integrates up to 7th or 9th order
n_quad_points = 4
s_points, s_weights = np.polynomial.legendre.leggauss(n_quad_points)
#sp = [-np.sqrt(3/7+(2/7)*np.sqrt(6/5)),-np.sqrt(3/7-(2/7)*np.sqrt(6/5)),np.sqrt(3/7-(2/7)*np.sqrt(6/5)),np.sqrt(3/7+(2/7)*np.sqrt(6/5))]
#sw = [(18-np.sqrt(30))/36,(18+np.sqrt(30))/36,(18+np.sqrt(30))/36,(18-np.sqrt(30))/36]
#print('points',s_points,sp)
# print('weights',s_weights,sw)
# Rescale points and weights for the [0, 1] interval sufl = ufl.variable(ufl.real, name='s')
sufl = fd.Constant(0.0)
si = (s_points+1)/2
wi = s_weights/2
hs = h0 + sufl*(hn-h0) 
phis = phi0 + sufl*(phin-phi0)
qs = q0 + sufl*(qn-q0)
ubars = fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs)
print('wi, si',wi,si)

# METHOD-3 checked quad-4: print('HALLOOOOOO', 2*si[0]-1, 2*wi[0], 2*si[1]-1, 2*wi[1], 2*si[3]-1, 2*wi[3])
suf = si[0]
hs1 = h0 + suf*(hn-h0) 
phis1 = phi0 + suf*(phin-phi0)
qs1 = q0 + suf*(qn-q0)
ubars1 = fd.grad(phis1)+hs1*qs1*fd.grad(hs1)+(1/3)*hs1**2*fd.grad(qs1)
suf = si[1]
hs2 = h0 + suf*(hn-h0) 
phis2 = phi0 + suf*(phin-phi0)
qs2 = q0 + suf*(qn-q0)
ubars2 = fd.grad(phis2)+hs2*qs2*fd.grad(hs2)+(1/3)*hs2**2*fd.grad(qs2)
suf = si[2]
hs3 = h0 + suf*(hn-h0) 
phis3 = phi0 + suf*(phin-phi0)
qs3 = q0 + suf*(qn-q0)
ubars3 = fd.grad(phis3)+hs3*qs3*fd.grad(hs3)+(1/3)*hs3**2*fd.grad(qs3)
suf = si[3]
hs4 = h0 + suf*(hn-h0) 
phis4 = phi0 + suf*(phin-phi0)
qs4 = q0 + suf*(qn-q0)
ubars4 = fd.grad(phis4)+hs4*qs4*fd.grad(hs4)+(1/3)*hs4**2*fd.grad(qs4)
H_s1 = ( 0.5*hs1*fd.inner(fd.grad(phis1)+ah01*hs1*qs1*fd.grad(hs1)+(aq01*1/3)*hs1**2*fd.grad(qs1),fd.grad(phis1)+ah01*hs1*qs1*fd.grad(hs1)+(aq01*1/3)*hs1**2*fd.grad(qs1))+aq02*(1/6)*hs1**3*qs1**2+0.5*grav*hs1**2-grav*hs1*H0+0.5*grav*H0**2+(betaa/90)*hs1**5*fd.inner(fd.grad(qs1),fd.grad(qs1)) )
H_s2 = ( 0.5*hs2*fd.inner(fd.grad(phis2)+ah01*hs2*qs2*fd.grad(hs2)+(aq01*1/3)*hs2**2*fd.grad(qs2),fd.grad(phis2)+ah01*hs2*qs2*fd.grad(hs2)+(aq01*1/3)*hs2**2*fd.grad(qs2))+aq02*(1/6)*hs2**3*qs2**2+0.5*grav*hs2**2-grav*hs2*H0+0.5*grav*H0**2+(betaa/90)*hs2**5*fd.inner(fd.grad(qs2),fd.grad(qs2)) )
H_s3 = ( 0.5*hs3*fd.inner(fd.grad(phis3)+ah01*hs3*qs3*fd.grad(hs3)+(aq01*1/3)*hs3**2*fd.grad(qs3),fd.grad(phis3)+ah01*hs3*qs3*fd.grad(hs3)+(aq01*1/3)*hs3**2*fd.grad(qs3))+aq02*(1/6)*hs3**3*qs3**2+0.5*grav*hs3**2-grav*hs3*H0+0.5*grav*H0**2+(betaa/90)*hs3**5*fd.inner(fd.grad(qs3),fd.grad(qs3)) )
H_s4 = ( 0.5*hs4*fd.inner(fd.grad(phis4)+ah01*hs4*qs4*fd.grad(hs4)+(aq01*1/3)*hs4**2*fd.grad(qs4),fd.grad(phis4)+ah01*hs4*qs4*fd.grad(hs4)+(aq01*1/3)*hs4**2*fd.grad(qs4))+aq02*(1/6)*hs4**3*qs4**2+0.5*grav*hs4**2-grav*hs4*H0+0.5*grav*H0**2+(betaa/90)*hs4**5*fd.inner(fd.grad(qs4),fd.grad(qs4)) )

nmeth = 0
# METHOD-1 Define Hamiltonian; integrate via quadrature
# H_s = ( 0.5*hs*fd.inner(ubars,ubars)+(1/6)*hs**3*qs**2+0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )
H_s = ( 0.5*hs*fd.inner(fd.grad(phis)+ah01*hs*qs*fd.grad(hs)+aq01*(1/3)*hs**2*fd.grad(qs),\
                        fd.grad(phis)+ah01*hs*qs*fd.grad(hs)+aq01*(1/3)*hs**2*fd.grad(qs))\
        +aq02*(1/6)*hs**3*qs**2\
        +0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )
#  s-Integrate the Hamiltonian density: loop evaluates Hamiltonian at each Gaussian point and sums weighted results
H_integrated = (fd.Constant(0.0)*hs)*fd.dx(degree=vpolyp)
H_s_evaluated = []
for ii in range(n_quad_points):
    # Replace symbolic 's_ufl' with numeric 'si' value for iteration: old: H_s_evaluated = fd.replace(H_s, {sufl: si[ii]}) # OLD
    H_s_evaluated.append(fd.replace(H_s, {sufl: si[ii]}))
    H_integrated += wi[ii]*H_s_evaluated[ii]*fd.dx(degree=vpolyp)  # Accumulate weighted Hamiltonian value
# s-integrated Hamiltonian is a UFL Form object:
Hamiltonian = H_integrated # *fd.dx(degree=vpolyp) # METHOD-1 OLD METHD-1
if nmeth==3:
    Hamiltonian =  (wi[0]*H_s1+wi[1]*H_s2+wi[2]*H_s3+wi[3]*H_s4)*fd.dx(degree=vpolyp) # METHOD-3
delHamdelphi = fd.derivative(Hamiltonian,phi0,du=vvmp0)+fd.derivative(Hamiltonian,phin,du=vvmp0)
delHamdelh = fd.derivative(Hamiltonian,h0,du=vvmp1)+fd.derivative(Hamiltonian,hn,du=vvmp1)
delHamdelpsi = fd.derivative(Hamiltonian,q0,du=vvmp2)+fd.derivative(Hamiltonian,qn,du=vvmp2)


delHamdelphi = ( fd.inner( fd.grad(vvmp0),(1/6)*(2*h0*fd.grad(phi0)+2*hn*fd.grad(phin)+h0*fd.grad(phin)+hn*fd.grad(phi0)) ) )*fd.dx(degree=vpolyp)
delHamdelh = ( vvmp1*grav*(0.5*(hn+h0)-H0) + vvmp1*(1/6)*( fd.inner(fd.grad(phi0),fd.grad(phi0))+fd.inner(fd.grad(phin),fd.grad(phin))+fd.inner(fd.grad(phi0),fd.grad(phin))  )  )*fd.dx(degree=vpolyp)
delHamdelpsi = ( vvmp2*qn )*fd.dx(degree=vpolyp)

# METHOD-2 define three weak forms ito hs, phis, qs, either a) explicitly or b) via Hamiltonian, integrate via quadrature an then use that:
# PHI:
AVFhdelphi = ( hs*fd.inner(fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs),fd.grad(vvmp0)) )*fd.dx(degree=vpolyp)
# H:
AVFphidelh = ( hs*fd.inner(fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs),qs*fd.grad(hs)*vvmp1+qs*hs*fd.grad(vvmp1)+(2/3)*hs*fd.grad(qs)*vvmp1)+\
               ((1/2)*fd.inner( fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs), fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs))\
                +(1/2)*hs**2*qs**2+grav*(hs-H0)+(betaa/18)*hs**4*fd.inner(fd.grad(qs),fd.grad(qs)))*vvmp1 )*fd.dx(degree=vpolyp)

AVFphidelh = ( hs*fd.inner(ubars,qs*fd.grad(hs)*vvmp1+(2/3)*hs*fd.grad(qs)*vvmp1+qs*hs*fd.grad(vvmp1))+\
               ((1/2)*fd.inner(ubars, ubars)+\
                (1/2)*hs**2*qs**2+grav*(hs-H0)+(betaa/18)*hs**4*fd.inner(fd.grad(qs),fd.grad(qs)))*vvmp1 )*fd.dx(degree=vpolyp)
# PSI
AVFpsidelp = ( hs*fd.inner(fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs),hs*fd.grad(hs)*vvmp2+(1/3)*hs**2*fd.grad(vvmp2))+\
               (1/3)*hs**3*qs*vvmp2+(betaa/45)*hs**5*fd.inner(fd.grad(qs),fd.grad(vvmp2)) )*fd.dx(degree=vpolyp)
AVFhdelphiint = 0.0
AVFphidelhint = 0.0
AVFpsidelpint = 0.0
for ii in range(n_quad_points):
    AVFhdelphieval = fd.replace(AVFhdelphi, {sufl: si[ii]})
    AVFhdelphiint += wi[ii]*AVFhdelphieval
    AVFphidelheval = fd.replace(AVFphidelh, {sufl: si[ii]})
    AVFphidelhint += wi[ii]*AVFphidelheval 
    AVFpsidelpeval = fd.replace(AVFpsidelp, {sufl: si[ii]})
    AVFpsidelpint += wi[ii]*AVFpsidelpeval
# if nmeth==1: # explicit weak forms
delHamdelphi1 = AVFhdelphiint
delHamdelh1 = AVFphidelhint
delHamdelpsi1 = AVFpsidelpint
phi_expr1 = ( vvmp0*(hn-h0)/dt )*fd.dx(degree=vpolyp) - delHamdelphi1 # 
h_expr1 = ( vvmp1*((phin-phi0)/dt +ccinv*Forcingterm + grav*gravwm )  )*fd.dx(degree=vpolyp) + delHamdelh1
nq = 1
if nq==0:
    q_expr1 = ( vvmp2*qn )*fd.dx(degree=vpolyp) # +BC_qs_term
    q_expr = ( vvmp2*qn )*fd.dx(degree=vpolyp) # +BC_qs_term
else:
    q_expr1 = delHamdelpsi1 # ( vvmp1*qn )*fd.dx(degree=vpolyp)
    q_expr = delHamdelpsi # +BC_qs_term
    
# if nmeth==1: derived weak forms
phi_expr = ( vvmp0*(hn-h0)/dt )*fd.dx(degree=vpolyp) - delHamdelphi # 
h_expr = ( vvmp1*((phin-phi0)/dt +ccinv*Forcingterm + grav*gravwm )  )*fd.dx(degree=vpolyp) + delHamdelh


phi_expr = ( vvmp0*(hn-h0)/dt \
             -fd.inner( fd.grad(vvmp0),(1/6)*(2*h0*fd.grad(phi0)+2*hn*fd.grad(phin)+h0*fd.grad(phin)+hn*fd.grad(phi0)) ) )*fd.dx(degree=vpolyp) # nil/rest solution
h_expr   = (  vvmp1*(phin-phi0)/dt \
              +vvmp1*(1/6)*( fd.inner(fd.grad(phi0),fd.grad(phi0))+fd.inner(fd.grad(phin),fd.grad(phin))+fd.inner(fd.grad(phi0),fd.grad(phin))  ) \
              +vvmp1*(grav*(0.5*(hn+h0)-H0)+ccinv*Forcingterm + grav*gravwm )  )*fd.dx(degree=vpolyp)



if nwave==1:
    Z_expr   = ( vvmp3*( Area*meff*(Wn-W0)/dt+Area*meff*grav-ccinv*Forcingterm ))*fd.dx(degree=vpolyp)
    W_expr   = ( vvmp4*Area*( (Zn-Z0)/dt-0.5*(Wn+W0) ))*fd.dx(degree=vpolyp)   # 
else:
    Z_expr   = ( vvmp3*( Zn ))*fd.dx(degree=vpolyp)
    W_expr   = ( vvmp4*( Wn ))*fd.dx(degree=vpolyp)   # nil/rest solution
F_exprnl = phi_expr+h_expr+q_expr+Z_expr+W_expr
F_exprnl1 = phi_expr1+h_expr1+q_expr1+Z_expr+W_expr

# Action on current solution: this produces a scalar Form; assemble it to a number
act_diff_form = fd.action(F_exprnl1- F_exprnl, result_mixedmp)
act_diff = fd.assemble(act_diff_form)
print("Overall action difference (should be ~0):", float(act_diff))

# Nonsense test: Action on current solution (a scalar)
act_dH_q = fd.assemble(fd.action(delHamdelpsi-delHamdelpsi1, result_mixedmp))
print("TEST1: action of integral dH/dvarphi (should be ~0):", float(act_dH_q))

if nmeth==1:
    F_exprnl = F_exprnl1


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
solver_parameters = {
    'snes_type': 'newtonls',  # Use a line-search Newton's method
    'mat_type': 'nest',
    'pc_fieldsplit_0_fields': '0,1,2',  # Continuous fields h, phi, q
    'pc_fieldsplit_1_fields': '3,4',    # R variables Z, W
    'pc_fieldsplit_1': {
        'mat_type': 'nest',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    },
    'snes_rtol': 1.0e-12, # Relative tolerance
    'snes_atol': 1.0e-14, # Absolute tolerance
    'snes_monitor': None, # This will turn on the default SNES monitor
}

solver_parameters3 = {
    'snes_type': 'newton',  # Use a line-search Newton's method
    'mat_type': 'nest',
    'snes_max_it': 50,         # Maximum number of iterations
    'snes_rtol': 1.0e-14,      # Relative tolerance
    'snes_atol': 1.0e-12,      # Absolute tolerance
    'snes_monitor': None,      # Monitor the convergence
    # 'pc_type': 'lu',           # Use a direct solver
}


# Hydrostatic solver for initial condition: # wavebuoy_hydrotstaticnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_hyexprnl, result_mixedmp))
wavebuoy_hydrotstaticnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_hyexprnl, result_mixedmp), solver_parameters=solver_parameters4)
# Dynamic solver: wavebuoy_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_exprnl, result_mixedmp), solver_parameters=solver_parameters4)
#V_q =  mixed_Vmp.sub(2) # Function space for psi is 3rd component
#bc_psi = fd.DirichletBC(V_q, fd.Constant(0.0), "on_boundary") # Create a DirichletBC for psi=0 on all boundaries # Assumes all exterior boundaries are vertical walls
#V_qphi =  mixed_Vmp.sub(0) # Function space for phi is 1st component
#bc_phi = fd.DirichletBC(V_qphi, fd.Constant(0.0), "on_boundary") # Create a DirichletBC for psi=0 on all boundaries # Assumes all exterior boundaries are vertical walls
#V_phi = mixed_Vmp.sub(0)
#V_h = mixed_Vmp.sub(1)
#V_q = mixed_Vmp.sub(2)
#bc_expr = -phi0-((hn+h0)/2)**2*((qn+q0)/2) # Create the UFL expression for the boundary value -(1/2)*hn**2*qn
#bc_phi = fd.DirichletBC(V_phi, bc_expr, "on_boundary")
wavebuoy_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_exprnl, result_mixedmp)) # , solver_parameters=solver_parameters3)
#wavebuoy_nl.parameters['snes_rtol'] = 1.e-12 # does not work n that not gettibg used
#wavebuoy_nl.parameters['snes_atol'] = 1.e-14 # does not work n that not gettibg used


# lines_parameters = {'ksp_type': 'gmres', 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC', 'star_construct_dim': 2,'star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}

###### OUTPUT FILES and initial PLOTTING ##########
save_path =  "data/"
if not os.path.exists(save_path):
    os.makedirs(save_path) 
#outfile_phin = fd.File(save_path+"phin.pvd")
#outfile_phin = VTKFile(os.path.join(save_path, "phin.pvd"))
outfile_phin = os.path.join(save_path, "phin.pvd")
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
    solver_reason = wavebuoy_hydrotstaticnl.solve()
    phin, hn, qn, Zn, Wn = fd.split(result_mixedmp)
    if nmpi==0:
        print(f"Solving for cc = {cc_val}...")
        print(f"Solver for cc={cc_val} converged reason: {solver_reason}")
    elif nmp==1:
        PETSc.Sys.Print(f"Hydrostatic solve cc={cc_val}...")

        
# Plotting hydrostatic initial condition state
t = 0.0
ii = 0
Z1.interpolate(Zn)
W1.interpolate(Wn)
h1.interpolate(hn)
phi1.interpolate(phin)
q1.interpolate(qn)
Z00 = np.array(Z1.vector())
W00 = np.array(W1.vector())
if nmpi==0:
    plt.figure(2)
    plt.ion() 
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(2,2)
    fig, axes = plt.subplots(2, 2)
    # axes is a 2D array, so you can access each plot by its row and column index
    ax1 = axes[0, 0] # Top-left plot
    ax2 = axes[0, 1] # Top-right plot
    ax3 = axes[1, 0] # Bottom-left plot
    ax4 = axes[1, 1] # Bottom-right plot
    phi1vals = np.array([phi1.at(xslice,y) for y in yvals])
    h1vals = np.array([h1.at(xslice,y) for y in yvals])
    q1vals = np.array([q1.at(xslice,y) for y in yvals])
    ax1.plot(yvals,h1vals,'-')
    ax1.plot(yvals,0.0*h1vals,'-r')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(yvals,phi1vals,'-')
    ax3.plot(yvals,q1vals,'-')
    ax4_twin = ax4.twinx()
    ax4.plot(t, Z00, 'b.', label='Z(t)')
    #ax4.set_ylabel('Z(t)', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.plot(t, W00, 'rx', label='W(t)')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    # ax4_twin.set_ylabel('W(t)', color='r') ax4.legend() # Legend to distinguish Z and W
    ha1, la1 = ax4.get_legend_handles_labels()
    ha2, la2 = ax4_twin.get_legend_handles_labels() # Combine them
    ax4.legend(ha1 + ha2, la1 + la2, loc='best')
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
    ax3.set_ylabel(r'$\psi(\frac {1}{2} L_x,y,t)$ ',fontsize=size)
    ax3.set_xlabel(r'$y$ ',fontsize=size)
    ax4.set_xlabel(r'$t$ ',fontsize=size)
    
# Initial condition is hydrostatic rest state just calculated
phi0.assign(phi1)
h0.assign(h1)
q0.assign(q1)
W0.assign(W1)
Z0.assign(Z1)
# ubars = fd.grad(phis) + hs*qs*fd.grad(hs) + (1/3)*hs**2*fd.grad(qs)
# H_s = ( 0.5*hs*fd.inner(ubars,ubars)+(1/6)*hs**3*qs**2+0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )
# H_s = ( 0.5*hs*fd.inner(fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs),fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs))
#         +(1/6)*hs**3*qs**2+0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )

E0 = fd.assemble( ( 0.5*fd.inner( fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn), fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn)) \
                    +aq02*(1/6)*hn**3*qn**2+0.5*grav*hn**2-grav*hn*H0+0.5*grav*H0**2+(betaa/90)*hn**5*fd.inner(fd.grad(qn),fd.grad(qn)) \
                    +(nwave/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) + nwave*Area*meff*(0.5*Wn**2+grav*Zn) )*fd.dx(degree=vpolyp) )
if nmpi==0:
    print('IC E0:',E0,grav)
gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))

def monitor(snes, its, fnorm):
    PETSc.Sys.Print(f"  SNES it {its}, residual norm {fnorm:.4e}")
snes = wavebuoy_nl.snes

nplotyes=1
tic = tijd.time()
nstop = 0
E1 = E0
E00 = E0
# Time loop starts: Needs to start atm with Zn,Wn,phim,hn, qn of hydrostatic solve
while t <= 1.0*(t_end + dt):
    gravwmtime.assign(gravwavemakertime(t+0.5*dt,sigma,twmstop))
    wavebuoy_nl.solve()

    ii = ii+1
    t+= dt
    E00 = E1
    E1 = fd.assemble( (  0.5*fd.inner( fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn), fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn)) \
                         +aq02*(1/6)*hn**3*qn**2+0.5*grav*hn**2-grav*hn*H0+0.5*grav*H0**2+(betaa/90)*hn**5*fd.inner(fd.grad(qn),fd.grad(qn)) \
                         +(nwave/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) + nwave*Area*meff*(0.5*Wn**2+grav*Zn) )*fd.dx(degree=vpolyp) )
    
    phin, hn, qn, Zn, Wn = fd.split(result_mixedmp)

    Z1.interpolate(Zn)
    W1.interpolate(Wn)
    h1.interpolate(hn)
    phi1.interpolate(phin)
    q1.interpolate(qn)
    if (t in t_plot): # Plotting # print('ii, t',ii,t) # print('Plotting starts') # if nplotyes==1:

        fnorm = snes.getFunctionNorm()
        PETSc.Sys.Print(f"Step {ii}, final SNES residual norm {fnorm:.4e}", E00, E1)
        PETSc.Sys.Print(" Converged reason:", snes.getConvergedReason())
        
        if nmpi==0:
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
            # ubars = fd.grad(phis) + hs*qs*fd.grad(hs) + (1/3)*hs**2*fd.grad(qs)
            # H_s = ( 0.5*hs*fd.inner(ubars,ubars)+(1/6)*hs**3*qs**2+0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )
            # H_s = ( 0.5*hs*fd.inner(ubars,ubars)+(1/6)*hs**3*qs**2+0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )
            # H_s = ( 0.5*hs*fd.inner(fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs),fd.grad(phis)+hs*qs*fd.grad(hs)+(1/3)*hs**2*fd.grad(qs))
            #         +(1/6)*hs**3*qs**2+0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2+(betaa/90)*hs**5*fd.inner(fd.grad(qs),fd.grad(qs)) )

            E1 = fd.assemble( (  0.5*fd.inner( fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn), fd.grad(phin)+ah01*hn*qn*fd.grad(hn)+aq01*(1/3)*hn**2*fd.grad(qn)) \
                                 +aq02*(1/6)*hn**3*qn**2+0.5*grav*hn**2-grav*hn*H0+0.5*grav*H0**2+(betaa/90)*hn**5*fd.inner(fd.grad(qn),fd.grad(qn)) \
                                 +(nwave/cc**2)*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) + nwave*Area*meff*(0.5*Wn**2+grav*Zn) )*fd.dx(degree=vpolyp) )
            print('E0, E1, |E1-E0|/E0:',E0, E1, np.abs(E1-E0)/np.abs(E0))
            if t>twmstop and nstop==0:
                E0 = E1
                nstop = 1
                print('time, Twm E0=E1',t,twmstop,E0,E1)
            hbuoy0 = np.heaviside(yvLw-Ly+Lw,0.5)*(Z00-Keel-tanalp*(yvLw-Ly))
            ax1.plot(yvLw,hbuoy0,'--r') # Plot buoy shape
            if nmpi==0:
                plt.draw()
                plt.pause(0.0001)
    elif nmpi==1:
        PETSc.Sys.Print('Time:', t)
    phi0.assign(phi1) # Copy new states to old states for next time step
    h0.assign(h1)
    q0.assign(q1)
    W0.assign(W1)
    Z0.assign(Z1)
# End while/time loop
toc = tijd.time() - tic
#
if nprintout==1 & nmpi==0:
    print('Elapsed time (min):', toc/60)
else:
    PETSc.Sys.Print('Elapsed time (min):', toc/60)
# outfile_phin.close() # outfile_hn.close() # outfile_qn.close()
plt.savefig("figs/wavebuoyavfc2025bou.png")
#
if nmpi==0:
    plt.show()
if nprintout==1 and nmpi==0:
    print('*************** PROGRAM ENDS ******************')
else:
    PETSc.Sys.Print('*************** PROGRAM ENDS ******************')
