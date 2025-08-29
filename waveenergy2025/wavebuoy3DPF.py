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
    # from meshing import get_mesh #  Works
    from meshing import get_mesh # Works also
elif nmpi==1:
    from meshing import get_mesh
import ufl
from petsc4py import PETSc
from mpi4py import MPI

os.environ["OMP_NUM_THREADS"] = "1"

# op2.init()
# parameters["coffee"]["O2"] = False
# parameters in SI units REMIS error of polynomials
# water domain and discretisation parameters
# nic = "linearw" # choice initial condition
nic = "rest" # rest state start
nvpcase = "AVF" # AVF= VP time discretisation energy-conserving approach
nCG = 1     # order of CG. Choices: ["1","2","4"]
nCGvert = nCG
multiple=1    # a resolution factor in spatial domain.  1 is basic. 2 is dense. 
multiple2=1   # a resolution factor in temporal domain. 1 is basic. 2 is dense.
     
if nic=="rest": # rest test
    H0 = 0.15
    Lz = H0
    Hk = 0.04
    nx = 19 # 39
    ny = 201 # 401
    nz = 4
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
    mesh = fd.ExtrudedMesh(mesh2d, nz, layer_height=Lz/nz, extrusion_type='uniform',name='mesh') # 3D mesh    
elif nmpi==1: # Another failed GG
    # Use rank-based filename to avoid conflicts
    # Temporarily modify the mesh file creation to use rank-specific names
    # You'll need to modify your get_mesh function to accept a rank parameter
    # For now, try this simpler approach:
    for rank in range(COMM_WORLD.size):
        if COMM_WORLD.rank == rank:
            mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)[0]
            mesh = fd.ExtrudedMesh(mesh2d, nz, layer_height=Lz/nz, extrusion_type='uniform',name='mesh')
            print(f"Rank {rank}: Mesh created")
        COMM_WORLD.barrier()  # Wait for this rank to finish before next rank starts    # Force all processes to create mesh sequentially with a delay
    import time
    time.sleep(COMM_WORLD.rank * 0.1)  # Stagger the file access
    mesh2d = get_mesh(Lxx, Lyy, nxx+1, nyy, Lr)[0] 
    mesh = fd.ExtrudedMesh(mesh2d, nz, layer_height=Lz/nz, extrusion_type='uniform',name='mesh')
    print(f"Rank {COMM_WORLD.rank}: Mesh created")
   
        
if nmpi==0:
    print("Mesh generated successfully by rank 0.") # ADDED LINE: Print a confirmation message
    print(f"2D base mesh elements: {mesh2d.ufl_cell()}")
    print(f"3D extruded mesh elements: {mesh.ufl_cell()}")
elif nmpi==1:
    PETSc.Sys.Print('Mesh generated successfully')
    
nplotmesh = 0
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
x = mesh.coordinates
top_id = 'top'
seps = 0.0 #  -10**(-10)
yvals = np.linspace(0.0+seps, Ly-seps, ny)
xslice = 0.5*Lx
# 
t = 0
if nic=="rest":    
    ttilde = 50 # time units 
    t_end = 1.0 # 0.8*1.5
    Nt = 40
    dt = t_end/Nt #
    dtt = 1.2/Nt
    dt3 = np.minimum(Lx/nx,Ly/ny)/np.sqrt(grav*H0)
    CFL1 = 0.25*0.125
    CFL = 0.5
    dt = CFL*dt3
    print('time steps',dt,CFL1*dtt,CFL1*dt3,dt3)
    nplot = 40
    nplotZW = 40
    tijde = []
    while (t <= t_end+1*dt):
        tijde.append(t)
        t+= dt
    nt = int(len(tijde)/nplot)
    ntZW = int(len(tijde)/nplotZW)
    t_plot = tijde[0::nt]
    t_plotZW = tijde[0::ntZW]
                     
#__________________  Quadratures and define function spaces  __________________# orders = 2*nCG  # horizontal
orders = [2*nCG, 2*nCGvert]  # horizontal and vertical
quad_rules = []
for order in orders:
    fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), order)
    # Check: # print(fiat_rule.get_points())     # print(fiat_rule.get_weights())
    point_set = GaussLobattoLegendrePointSet(fiat_rule.get_points())
    quad_rule = QuadratureRule(point_set, fiat_rule.get_weights())
    quad_rules.append(quad_rule)
quad__rule = TensorProductQuadratureRule(quad_rules)

#horiz_elt = FiniteElement("CG", fd.quadrilateral, nCG)      # Q1 on quadrilatera
#vert_elt = FiniteElement("CG", fd.interval, nCGvert)       # CG1 on interval
#element_W = TensorProductElement(horiz_elt, vert_elt)      # For V_W (3D interior)
# For V_S: Tensor product with R element for surface functions in 3D space
# The R element should be defined on interval for extruded meshes
#R_elt = FiniteElement("R", fd.interval, 0)  # R (constant) on interval
#element_S = TensorProductElement(horiz_elt, R_elt)  # Q × R tensor product
# Create function spaces
#V_W = fd.FunctionSpace(mesh, element_W)  # 3D interior potential
#V_S = fd.FunctionSpace(mesh, element_S)  # Surface functions in 3D space (Q × R)
V_R = fd.FunctionSpace(mesh, 'R', 0)     # Global constants
V_C = fd.FunctionSpace(mesh, 'DG', 0)     # Global constants does not work yet with solvers
V_W = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='CG', vdegree=nCGvert) # 3D interior potential varpsi; can mix degrees in hor and vert
V_S = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='R', vdegree=0) # 2D surface psi and h
if nmpi==0:
    print(f"V_W element: {V_W.ufl_element()}")
    print(f"V_S element: {V_S.ufl_element()}")
    print(f"V_R element: {V_R.ufl_element()}")

psi0 = fd.Function(V_S, name="psi0") # velocity potential at level n at free surface
h0 = fd.Function(V_S, name="h0") # water depth at level n
varpsi0 = fd.Function(V_W, name="varpsi0") # q at level n
Z0 = fd.Function(V_R, name="Z0") # Z at level n
W0 = fd.Function(V_R, name="W0") # W at level n
psi1 = fd.Function(V_S, name="psi1") # velocity potential at level n+1 at free surface
h1 = fd.Function(V_S, name="h1") # water depth at level n+1
varpsi1 = fd.Function(V_W, name="varpsi1") # q at level n+1
Z1 = fd.Function(V_R, name="Z1") # Z at level n+1
W1 = fd.Function(V_R, name="W1") # W at level n+1
phihat = fd.Function(V_W)
if nvpcase=="AVF":
    mixed_Vmp = V_S * V_S * V_W * V_R * V_R
    result_mixedmp = fd.Function(mixed_Vmp)
    vvmp = fd.TestFunction(mixed_Vmp)
    vvmp0, vvmp1, vvmp2, vvmp3, vvmp4 = fd.split(vvmp) # Test functions, represent "blocks".
    psin, hn, varpsin, Zn, Wn = fd.split(result_mixedmp) # Variables
    BC_varphi_mixedmp = fd.DirichletBC(mixed_Vmp.sub(2), 0, top_id) # varphimp condition for modified midpoint

if nmpi==0:
    print(f"2D base mesh elements: {mesh2d.ufl_cell()}")
    print(f"3D extruded mesh elements: {mesh.ufl_cell()}")
    print(f"Function space element: {V_W.ufl_element()}")
    
# Initialise variables; projections on main variables at initial time
nphihatz = "Unity" # "Unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.
if nphihatz=="GLL1":
    fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), nCGvert+1) # GLL
    zk = (H0*fiat_rule.get_points())
    phihatexpr = fd.product( (x[2]-zk.item(kk))/(H0-zk.item(kk)) for kk in range(0,nCGvert-1,1) )
    phihat.interpolate(phihatexpr) # dphihat = phihat.dx(2) # May not work and in that case specify the entire product: dpsidxi3 = psimp*phihat.dx(1)
elif nphihatz=="Unity":
    phihat.assign = 1.0 # dphihat = phihat.dx(2)
vpolyp = 15
cc = fd.Constant(100)
ccinv = cc**(-1)
# VP or weak forms hydrostatic rest case fd.ds_t(degree=vpoly) 
phi_hyexpr = ( vvmp0*( psin ))*fd.ds_t(degree=vpolyp) # nil/rest solution
h_hyexpr   = ( vvmp1*( grav*(hn-H0)+ccinv*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) ))*fd.ds_t(degree=vpolyp)
q_hyexpr   = ( vvmp2*( varpsin ))*fd.dx(degree=vpolyp)   # nil/rest solution interior potential
Z_hyexpr   = ( vvmp3*( Area*Mm*grav-rho0*ccinv*fd.exp(-cc*(Zn-hn-Keel-tanalp*(x[1]-Ly))) ))*fd.ds_t(degree=vpolyp)
# Z_expr   = ( vvmp3*( Area*meff*(Wn-W0)/dt+Area*meff*grav-ccinv*Forcingterm ))*fd.ds_t(degree=vpolyp)
W_hyexpr   = ( vvmp4*( Wn ))*fd.ds_t(degree=vpolyp)   # nil/rest solution
F_hyexprnl = phi_hyexpr+h_hyexpr+q_hyexpr+Z_hyexpr+W_hyexpr


# VP or weak forms dynamic case: first test is rest state stays rest state; next test add forcing
Amp = 0.025 #  0.025
L1 = 0.2
Twm = 0.5 # 0.5
sigma = 2.0*np.pi/Twm
twmstop = 0.0*Twm # 0.0 no wave forcing; >0 wave forcing
gravwmtime = fd.Constant(0.0)
def gravwavemakertime(t,sigma,twmstop):
    if t<twmstop:
        return 0.0*np.sin(sigma*t)
    else:
        return 0.0
gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))
gravwm = Amp*fd.conditional(x[1]<L1,((x[1]-L1)/L1)*gravwmtime,0.0)
# 
nwave = 0 # 0: no buoy dynamics; 1: buoy dynamics
if nwave==1:
    deltacons = Zn-hn-Z0+h0
    Forcingterm = fd.exp(-0.5*cc*(Zn-hn+Z0-h0)+cc*(Keel+tanalp*(x[1]-Ly)))*fd.conditional(fd.eq(deltacons,0.0),1.0,-ccinv*( 2.0*fd.sinh(-0.5*cc*deltacons) )/deltacons)
else:
    Forcingterm = 0.0

# Define AVF s-integration scheme using Gaussian quadrature: 4-pnt or 5-pnt rules sufficient for 7th order polynomial: integrates up to 7th or 9th order
n_quad_points = 4
s_points, s_weights = np.polynomial.legendre.leggauss(n_quad_points)
sufl = fd.Constant(0.0)
si = (s_points+1)/2
wi = s_weights/2
hs = h0 + sufl*(hn-h0) 
psis = psi0 + sufl*(psin-psi0)
varphis = varpsi0 + sufl*(varpsin-varpsi0)
gradvarphis = fd.grad(varphis)
gradvarphisxy = fd.as_vector([gradvarphis[0], gradvarphis[1]])
gradpsis = fd.grad(psis)
gradpsisxy = fd.as_vector([gradpsis[0], gradpsis[1]])
gradhs = fd.grad(hs)
gradhsxy = fd.as_vector([gradhs[0], gradhs[1]])
# Polynomial part of Hamitonian:
Hsp = ( 0.5*grav*hs**2-grav*hs*H0+0.5*grav*H0**2 )*fd.ds_t(degree=vpolyp)+ \
    ( 0.5*(hs/H0)*fd.inner(phihat*gradpsisxy+gradvarphisxy,phihat*gradpsisxy+gradvarphisxy)+ \
      -(x[2]/H0)*fd.inner(phihat*gradpsisxy+gradvarphisxy, gradhsxy*(psis*phihat.dx(2)+varphis.dx(2))) )*fd.dx(degree=vpolyp)
# Two terms kinetic energy part of Hamitonian proportional to 1/h   
a0 = (psi0*phihat.dx(2)+varpsi0.dx(2))
an = psin*phihat.dx(2)+varpsin.dx(2)-a0
anma0 = (psin*phihat.dx(2)+varpsin.dx(2)-psi0*phihat.dx(2)+varpsi0.dx(2))
kn = hn-h0
#print('a0 inner itself:', fd.assemble(fd.inner(a0,a0)*fd.dx))
#print('an inner itself:', fd.assemble(fd.inner(an,an)*fd.dx))  
#print('a0 inner an:', fd.assemble(fd.inner(a0,an)*fd.dx))
#
a02 = fd.inner(a0,a0)
an2 = fd.inner(an,an)
kn2 = fd.inner(kn,kn)
kn3 = fd.inner(kn2,kn)
Hs3h = (-2*an2*h0*kn + an*kn2*(4*a0 + an) + 2*(a0*kn - an*h0)**2*fd.ln(1+kn/h0) )/(2*kn3)
# Hs3h = (a0**2/kn)*fd.ln(1+kn/h0) + 2*(a0*an/kn**2)*(kn - h0*fd.ln(1+kn/h0)) + (an**2/(2*kn**3))*(-2*h0*kn + kn**2 + 2*h0**2*fd.ln(1+kn/h0))
Hs3hlH = (a02+a0*an+an2/3)/((hn+h0)/2)
#Hs3hlH = (fd.inner((psi0*phihat.dx(2)+varpsi0.dx(2)),(psi0*phihat.dx(2)+varpsi0.dx(2)))+fd.inner((psi0*phihat.dx(2)+varpsi0.dx(2)),(psin*phihat.dx(2)+varpsin.dx(2)-psi0*phihat.dx(2)+varpsi0.dx(2)))+fd.inner((psin*phihat.dx(2)+varpsin.dx(2)-psi0*phihat.dx(2)+varpsi0.dx(2)),(psin*phihat.dx(2)+varpsin.dx(2)-psi0*phihat.dx(2)+varpsi0.dx(2)))/3)/((hn))
Hs3h = ( 0.5*H0*fd.conditional(fd.eq(hn,h0), Hs3hlH, Hs3h) )*fd.dx(degree=vpolyp)
# Hs3h = ( hn )*fd.dx(degree=vpolyp)

gradh0 = fd.grad(h0)
gradh0xy = fd.as_vector([gradh0[0], gradh0[1]])
gradhn = fd.grad(hn)
gradhnxy = fd.as_vector([gradhn[0], gradhn[1]])
c0 = gradh0xy
cn = gradhnxy-gradh0xy
c0cn = fd.inner(c0,cn)
c02 = fd.inner(c0,c0)
cn2 = fd.inner(cn,cn)
e0 = a0
e02 = fd.inner(e0,e0)
en = an
en2 = fd.inner(en,en)
Hs4h = ( -12*cn2*en2*h0**3*kn + 6*en*h0**2*kn**2*(4*c0cn*en + 4*cn2*e0 + cn2*en) - 4*h0*kn**3*(3*c02*en2 + 12*c0cn*e0*en + 3*c0cn*en2 + 3*cn2*e02 + 3*cn2*e0*en + cn2*en2) +\
         kn**4*(24*c02*e0*en + 6*c02*en2 + 24*c0cn*e02 + 24*c0cn*e0*en + 8*c0cn*en2 + 6*cn2*e02 + 8*cn2*e0*en + 3*cn2*en2) +\
         12*(c0*kn - cn*h0)**2*(e0*kn - en*h0)**2*(fd.ln(1+kn/h0)))/(12*kn**5)
Hs4hlH = (12*c02*e02 + 12*c02*e0*en + 4*c02*en2 + 12*c0cn*e02 + 16*c0cn*e0*en + 6*c0cn*en2 + 4*cn2*e02 + 6*cn2*e0*en + 12*cn2*en2/5)/(6*(h0+hn))
Hs4h = ( 0.5*(x[2]**2/H0)*fd.conditional(fd.eq(hn,h0), Hs4hlH, Hs4h) ) *fd.dx(degree=vpolyp) 

#  s-Integrate the Hamiltonian density Hsp: loop evaluates Hamiltonian at each Gaussian point and sums weighted results
H_integrated = (fd.Constant(0.0)*hs)*fd.dx(degree=vpolyp)
Hs_evaluated = []
for ii in range(n_quad_points):
    # Replace symbolic 's_ufl' with numeric 'si' value for iteration:
    Hs_evaluated.append(fd.replace(Hsp, {sufl: si[ii]}))
    H_integrated += wi[ii]*Hs_evaluated[ii]  # Accumulate weighted Hamiltonian value
# s-integrated Hamiltonian is a UFL Form object:
# Hamiltonian = H_integrated + Hs3h + Hs4h # Accumulate weighted Hamiltonian value
Hamiltonian = H_integrated+ Hs3h + Hs4h  # Accumulate weighted Hamiltonian value
delHamdelphi = fd.derivative(Hamiltonian,psi0,du=vvmp0)+fd.derivative(Hamiltonian,psin,du=vvmp0)
delHamdelh = fd.derivative(Hamiltonian,h0,du=vvmp1)+fd.derivative(Hamiltonian,hn,du=vvmp1)
delHamdelpsi = fd.derivative(Hamiltonian,varpsi0,du=vvmp2)+fd.derivative(Hamiltonian,varpsin,du=vvmp2)

# Test with a simple constant test_integral = fd.assemble((fd.Constant(1.0)+fd.Constant(1.0)*hn)*fd.dx(degree=vpolyp)) print('ONNO Constant 1 integral:', test_integral/H0,Area)

phi_expr = ( fd.inner(vvmp0,(hn-h0)/dt) )*fd.ds_t(degree=vpolyp) - delHamdelphi # 
h_expr = ( fd.inner(vvmp1,((psin-psi0)/dt +ccinv*Forcingterm + grav*gravwm ))  )*fd.ds_t(degree=vpolyp) + delHamdelh
q_expr = delHamdelpsi

if nwave==1:
    Z_expr   = ( fd.inner(vvmp3,( Area*meff*(Wn-W0)/dt+Area*meff*grav-ccinv*Forcingterm)) )*fd.ds_t(degree=vpolyp)
    W_expr   = ( fd.inner(vvmp4,Area*((Zn-Z0)/dt-0.5*(Wn+W0))) )*fd.ds_t(degree=vpolyp)   #
else:
    Z_expr   = ( fd.inner(vvmp3,Zn) )*fd.ds_t(degree=vpolyp)
    W_expr   = ( fd.inner(vvmp4,Wn) )*fd.ds_t(degree=vpolyp)   # nil/rest solution
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
    'mat_type': 'nest',  # 'mat_type': 'nest',
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
        'mat_type': 'nest',  # 'mat_type': 'nest',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
9    },
}

solver_parameters3 = {
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

solver_parameters = {
    'snes_type': 'vinewtonrsls',  # Use a line-search Newton's method
    'mat_type': 'nest',
    'snes_max_it': 50,         # Maximum number of iterations
    'snes_rtol': 1.0e-10,      # Relative tolerance
    'snes_atol': 1.0e-12,      # Absolute tolerance
    'snes_monitor': None,      # Monitor the convergence
    'pc_type': 'lu',           # Use a direct solver
}

solver_parameters11 = {'mat_type': 'nest','ksp_type': 'gmres','pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC', 'star_construct_dim': 2,
                    'star_sub_sub_pc_type': 'lu','sub_sub_pc_factor_mat_ordering_type': 'rcm', 'snes_rtol': 1.0e-10,'snes_atol': 1.0e-12,'snes_monitor': None,}

# Hydrostatic solver for initial condition: # wavebuoy_hydrotstaticnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_hyexprnl, result_mixedmp))
wavebuoy_hydrotstaticnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_hyexprnl, result_mixedmp, bcs=BC_varphi_mixedmp), solver_parameters=solver_parameters4)
# Dynamic solver: wavebuoy_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_exprnl, result_mixedmp), solver_parameters=solver_parameters4)
wavebuoy_nl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(F_exprnl, result_mixedmp, bcs=BC_varphi_mixedmp)) # ,solver_parameters=solver_parameters)

# lines_parameters = {'ksp_type': 'gmres', 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC', 'star_construct_dim': 2,'star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}

###### OUTPUT FILES and initial PLOTTING ##########
save_path =  "data/"
if not os.path.exists(save_path):
    os.makedirs(save_path) 
#outfile_phn = fd.File(save_path+"phin.pvd")
#outfile_phin = VTKFile(os.path.join(save_path, "phin.pvd"))
outfile_phin = os.path.join(save_path, "phin.pvd")
#outfile_hn = fd.File(save_path+"hn.pvd") #outfile_qn = fd.File(save_path+"qn.pvd")
fileE = save_path+"potflow3denergy.txt"
# outputE = open(fileE,"w") # outputEtamax = open(filEtaMax,"w") # outfile_height.write(h_old, time=t)
# outfile_psi.write(psi_f, time=t) # outfile_varphi.write(varphi, time=t)

# Solve hydrostatic rest state:
cc_values = [100.0, 500.0, 1000.0, 2000.0, 4000.0] # , 4000.0, 5000.0, 7000.0] # , 7000.0, 10000.0]
# --- Continuation Loop Execution ---
if nmpi==0:
    print("Starting continuation loop...")
for cc_val in cc_values:
    # Update the Firedrake Constant with the new cc value
    cc.assign(cc_val)
    solver_reason = wavebuoy_hydrotstaticnl.solve()
    psin, hn, varpsin, Zn, Wn = fd.split(result_mixedmp) # Variables
    if nmpi==0:
        print(f"Solving for cc = {cc_val}...")
        print(f"Solver for cc={cc_val} converged reason: {solver_reason}")
    elif nmp==1:
        PETSc.Sys.Print(f"Hydrostatic solve cc={cc_val}...")

if nmpi==0:
    print(' Hallo Onno done hydrostatic; dt', dt)
elif nmp==1:
    PETSc.Sys.Print(f"Hallo Onno; hydrostatic solve done.")

    
# Plotting hydrostatic initial condition state
t = 0.0
ii = 0
Zn_actual = result_mixedmp.subfunctions[3]  # Zn is the 4th component (index 3) This one fails: Z1.interpolate(Zn)
Z1.project(Zn_actual) 
Wn_actual = result_mixedmp.subfunctions[4]  # Wn is the 5th component (index 4) W1.interpolate(Wn)
W1.project(Wn_actual)
hn_actual = result_mixedmp.subfunctions[1]  # hn is the 2nd component (index 1) h1.interpolate(hn)
h1.project(hn_actual)
psin_actual = result_mixedmp.subfunctions[0]  # psinn is the 1st component (index 0) psi1.interpolate(psin)
psi1.project(psin_actual)
varpsin_actual = result_mixedmp.subfunctions[2]  # varpsin is 3rd component (index 2) varpsi1.interpolate(varpsin)
varpsi1.project(varpsin_actual)
Z00 = np.array(Z1.vector())
W00 = np.array(W1.vector())

if nmpi==0:
    print(' Hallo Onno done hydrostatic 2') 
    plt.figure(2)
    plt.ion() 
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(2,2)
    fig, axes = plt.subplots(2, 2)
    # axes is a 2D array, so you can access each plot by its row and column index
    ax1 = axes[0, 0] # Top-left plot
    ax2 = axes[0, 1] # Top-right plot
    ax3 = axes[1, 0] # Bottom-left plot
    ax4 = axes[1, 1] # Bottom-right plot
    phi1vals = np.array([psi1.at(xslice,y,H0) for y in yvals])  # Along centreline of wave tank
    h1vals = np.array([h1.at(xslice,y,H0) for y in yvals])
    q1vals = np.array([varpsi1.at(xslice,y,H0) for y in yvals])
    ax1.plot(yvals,h1vals,'-')
    ax1.plot(yvals,0.0*h1vals,'-r')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(yvals,phi1vals,'-')
    ax3.plot(yvals,q1vals,'-')
    ax4_twin = ax4.twinx()
    ax4.plot(t, Z00, 'b.', label='Z(t)') #ax4.set_ylabel('Z(t)', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.plot(t, W00, 'rx', label='W(t)')
    ax4_twin.tick_params(axis='y', labelcolor='r') # ax4_twin.set_ylabel('W(t)', color='r') ax4.legend() # Legend to distinguish Z and W
    ha1, la1 = ax4.get_legend_handles_labels()
    ha2, la2 = ax4_twin.get_legend_handles_labels() # Combine them
    ax4.legend(ha1 + ha2, la1 + la2, loc='best')
    # Plot buoy shape:
    Lw = 0.1*Ly
    yvLw = np.linspace(Ly-Lw+10**(-10), Ly-10**(-10), ny)
    hbuoy0 = np.heaviside(yvLw-Ly+Lw,0.5)*(Z00-Keel-tanalp*(yvLw-Ly))
    ax1.plot(yvLw,hbuoy0,'--r')
    tsize2 = 12 # font size of image axes
    size = 10   # font size of image axes # ax1.set_xticklabels([]) # removed
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
psi0.project(psi1)
h0.project(h1)
varpsi0.project(varpsi1)
W0.project(W1)
Z0.project(Z1)
E0 = fd.assemble( ( 0.5*grav*h0**2-grav*h0*H0+0.5*grav*H0**2 + (nwave/cc**2)*fd.exp(-cc*(Z0-h0-Keel-tanalp*(x[1]-Ly))) +\
                    nwave*Area*meff*(0.5*W0**2+grav*Z0) )*fd.ds_t(degree=vpolyp)+ \
                  (0.5*(h0/H0)*fd.inner(phihat*fd.grad(psi0)+fd.grad(varpsi0)-(x[2]/h0)*fd.grad(h0)*(psi0*phihat.dx(2)+varpsi0.dx(2)),\
                                        phihat*fd.grad(psi0)+fd.grad(varpsi0)-(x[2]/h0)*fd.grad(h0)*(psi0*phihat.dx(2)+varpsi0.dx(2)))+\
                   0.5*(H0/h0)*(psi0*phihat.dx(2)+varpsi0.dx(2))**2 )*fd.dx(degree=vpolyp) )
if nmpi==0:
    print('IC E0:',E0,grav)
print('E0:',E0)
gravwmtime.assign(gravwavemakertime(t,sigma,twmstop))

nplotyes=1
tic = tijd.time()
nstop = 0
# Time loop starts: Needs to start atm with Zn,Wn,psin,hn,varpsin of hydrostatic solve
E1 = E0
E00 = E1 
while t <= 1.0*(t_end + dt):
    gravwmtime.assign(gravwavemakertime(t+0.5*dt,sigma,twmstop))
    if nmpi==0:
        print('Hallo tijd:',t)
        Hsp_val = fd.assemble(Hsp)
        print('Hsp assembled:', Hsp_val)
        delHamdelphi_val = fd.assemble(delHamdelphi)
        delHamdelh_val = fd.assemble(delHamdelh)
        delHamdelpsi_val = fd.assemble(delHamdelpsi)
        H_integrated_val = fd.assemble(H_integrated)
        Hs3h_val = fd.assemble(Hs3h)
        Hs4h_val = fd.assemble(Hs4h)

        # Extract numerical values from cofunctions
        delHamdelphi_num = delHamdelphi_val.dat.data
        delHamdelh_num = delHamdelh_val.dat.data
        delHamdelpsi_num = delHamdelpsi_val.dat.data
    
        print('delHamdelphi (should drive dh/dt):', delHamdelphi_num)
        print('delHamdelh (should drive dpsi/dt):', delHamdelh_num) 
        print('delHamdelpsi (should be 0):', delHamdelpsi_num)
        
        print('H_integrated:', H_integrated_val)
        print('Hs3h:', Hs3h_val)
        print('Hs4h:', Hs4h_val)
        
    wavebuoy_nl.solve()
    psin, hn, varpsin, Zn, Wn = fd.split(result_mixedmp) # Variables
    Zn_actual = result_mixedmp.subfunctions[3]  # Zn is the 4th component (index 3) This one fails: Z1.interpolate(Zn)
    Z1.project(Zn_actual) 
    Wn_actual = result_mixedmp.subfunctions[4]  # Wn is the 5th component (index 4) W1.interpolate(Wn)
    W1.project(Wn_actual)
    hn_actual = result_mixedmp.subfunctions[1]  # hn is the 2nd component (index 1) h1.interpolate(hn)
    h1.project(hn_actual)
    psin_actual = result_mixedmp.subfunctions[0]  # psinn is the 1st component (index 0) psi1.interpolate(psin)
    psi1.project(psin_actual)
    varpsin_actual = result_mixedmp.subfunctions[2]  # varpsin is 3rd component (index 2) varpsi1.interpolate(varpsin)
    varpsi1.project(varpsin_actual)

    E00 = E1
    E1 =  fd.assemble( ( 0.5*grav*h1**2-grav*h1*H0+0.5*grav*H0**2 + (nwave/cc**2)*fd.exp(-cc*(Z1-h1-Keel-tanalp*(x[1]-Ly))) +\
                         nwave*Area*meff*(0.5*W0**2+grav*Z0) )*fd.ds_t(degree=vpolyp)+ \
                       (0.5*(h1/H0)*fd.inner(phihat*fd.grad(psi1)+fd.grad(varpsi1)-(x[2]/h1)*fd.grad(h1)*(psi1*phihat.dx(2)+varpsi1.dx(2)),\
                                             phihat*fd.grad(psi1)+fd.grad(varpsi1)-(x[2]/h1)*fd.grad(h1)*(psi1*phihat.dx(2)+varpsi1.dx(2)))+\
                        0.5*(H0/h1)*(psi1*phihat.dx(2)+varpsi1.dx(2))**2 )*fd.dx(degree=vpolyp) )
    print('E00, E1, E1-E00', E00, E1, E1-E00)
    
    ii = ii+1
    t+= dt
    # OLD: Z1.interpolate(Zn) W1.interpolate(Wn) h1.interpolate(hn) phi1.interpolate(phin) q1.interpolate(qn)
    if (t in t_plot): # Plotting # print('ii, t',ii,t) # print('Plotting starts') # if nplotyes==1:
        if nmpi==0:
            print('ii, t',ii,t)
            Z00 = np.array(Z1.vector())
            W00 = np.array(W1.vector())
            phi1vals = np.array([psi1.at(xslice,y,H0) for y in yvals])
            h1vals = np.array([h1.at(xslice,y,H0) for y in yvals])
            q1vals = np.array([varpsi1.at(xslice,y,H0) for y in yvals])
            ax1.plot(yvals,h1vals,'-')
            ax1.plot(yvals,0.0*h1vals,'-r')
            ax2_twin.plot(yvals,phi1vals,'-')
            ax3.plot(yvals,q1vals,'-')
            ax4.plot(t, Z00, 'b.', label='Z(t)')
            ax4_twin.plot(t, W00, 'rx', label='W(t)')
            E1 =  fd.assemble( ( 0.5*grav*h1**2-grav*h1*H0+0.5*grav*H0**2 + (nwave/cc**2)*fd.exp(-cc*(Z1-h1-Keel-tanalp*(x[1]-Ly))) +\
                                 nwave*Area*meff*(0.5*W0**2+grav*Z0) )*fd.ds_t(degree=vpolyp)+ \
                  (0.5*(h1/H0)*fd.inner(phihat*fd.grad(psi1)+fd.grad(varpsi1)-(x[2]/h1)*fd.grad(h1)*(psi1*phihat.dx(2)+varpsi1.dx(2)),\
                   phihat*fd.grad(psi1)+fd.grad(varpsi1)-(x[2]/h1)*fd.grad(h1)*(psi1*phihat.dx(2)+varpsi1.dx(2)))+\
                   0.5*(H0/h1)*(psi1*phihat.dx(2)+varpsi1.dx(2))**2 )*fd.dx(degree=vpolyp) )
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
    psi0.project(psi1) # Copy new states to old states for next time step
    h0.project(h1)
    varpsi0.project(varpsi1)
    W0.project(W1)
    Z0.project(Z1)
# End while/time loop
toc = tijd.time() - tic
#
if nprintout==1 & nmpi==0:
    print('Elapsed time (min):', toc/60)
else:
    PETSc.Sys.Print('Elapsed time (min):', toc/60)
# outfile_phin.close() # outfile_hn.close() # outfile_qn.close()
#
if nmpi==0:
    plt.savefig("figs/wavebuoyavfc2025PF3.png")
    plt.show()
if nprintout==1 and nmpi==0:
    print('*************** PROGRAM ENDS ******************')
else:
    PETSc.Sys.Print('*************** PROGRAM ENDS ******************')
