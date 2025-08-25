from firedrake import mesh, COMM_WORLD, cython
import numpy as np
import os

#%% create_grid
def create_grid(Lx, Ly, Nx, Ny, d=None, Lc=None, Lr=None, Lb=None):
    '''
    Writes a file containing the (x,y) coordinates of the nodes used to form a
    quadrilateral mesh for a tank with a V-shaped contraction.
    
    An optional argument Lb can be passed, which denotes the position of the
    waterline of a buoy placed in the contraction. If Lb is passed, one of the
    straight lateral lines in the contraction is fixed to lie on the waterline,
    yielding differing left- and right- y-direction grid resolutions; the line
    for which the ratio between these resolutions is closest to 1 is chosen.
    
    Each node is represented by a unique number between 0 and Nn-1, where
    Nn = (Nx+1)*(Ny+1) + (Nx+2)*Nx/2 is the total number of nodes. The total
    number of elements is Nx*Ny + (Nx+1)*Nx/2, and the total number of 
    connections is Nx*(Ny+1)+(Nx+1)*Ny + (2*Nx+3)*Nx/2.
    
    Each line of the file corresponds to a unique element. The first 8 columns
    contain the (x,y) coordinates (eg x0,y0,x1,y1,x2,y2,x3,y3) and the last 4
    contain the numbers (eg 0,1,2,3) that define the element, given in an
    anti-clockwise order starting from the top-/top-left-most node.
    
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
    Lr : float
        The length of the rectangular part.
    Nn : int
        The total number of nodes.
    dycl : float
        The left resolution in the contraction
    dycr : float
        The right resolution in the contraction
    '''
    if not all(v == None or v > 0 for v in [Lx, Ly, Nx, Ny, d, Lc, Lr, Lb]):
        raise ValueError('All of the inputs must be positive')
    
    if not all(isinstance(v, int) for v in [Nx, Ny]):
        raise ValueError('Nx and Ny must both be integers')
    
    if Nx%2 != 0:
        raise ValueError('Nx must be even')
    
    if [d, Lc, Lr].count(None) != 2:
        raise ValueError('Exactly one of d, Lc or Lr must be given')
    
    if d is not None:
        if d <= Lx/2:
            raise ValueError('The chosen dimensions are not geometrically possible; the contraction is not closed')
        if d**2 >= (Lx/2)**2 + Ly**2:
            raise ValueError('The chosen dimensions are not geometrically possible; the entire tank is shorter than the contraction')
        Lc = (d**2 - (Lx/2)**2)**0.5
        Lr = Ly - Lc
    elif Lc is not None:
        if Lc >= Ly:
            raise ValueError('The chosen dimensions are not geometrically possible; the entire tank is shorter than the contraction')
        Lr = Ly - Lc
    elif Lr is not None:
        if Lr >= Ly:
            raise ValueError('The chosen dimensions are not geometrically possible; the entire tank is shorter than the rectangular part')
        Lc = Ly - Lr
    
    if Lb is not None:
        if Lb >= Ly:
            raise ValueError('The given waterline is not geometrically possible; it is not in the tank!')
        if Lr > Lb:
            raise ValueError('The given waterline is not geometrically possible; it is in the rectangular part')
        if Nx < 4:
            raise ValueError('Nx must be at-least 4 when a waterline exists')
    
    D = len(str(int((Nx+1)*(Ny+1) + (Nx+2)*Nx/2 - 1))) + 1
    
    mesh_data = open(f'mesh_Lx={Lx}_Ly={Ly}_Lr={Lr:.3f}_Nx={Nx}_Ny={Ny}_Lb={Lb}.txt', 'w')
    
    ### Rectangular region
    Nk = Nx*Ny          # number of elements in total
    
    dx = Lx/Nx          # grid resolution in x direction
    dy = Lr/Ny          # grid resolution in y direction
    
    for k in range(Nk):
        col = k//Nx
        row = k%Nx
        
        xT = row*dx
        xB = xT + dx
        
        yL = col*dy
        yR = yL + dy
        
        n1 = col*(Nx+1) + row
        n2 = n1 + 1
        n3 = n2 + Nx+1
        n4 = n3 - 1
        
        mesh_data.write(f'{xT:<10f} {yL:<10f} {xB:<10f} {yL:<10f} {xB:<10f} {yR:<10f} {xT:<10f} {yR:<10f} {n1:{D}d} {n2:{D}d} {n3:{D}d} {n4:{D}d}\n')
    
    ### Contraction region - Nx columns of elements
    Nn = (Nx+1)*Ny # no. of first node in region
    
    if Lb is None: # no waterline
        left = None # None = no waterline, True = left-of-waterline, False = right-of-waterline
        Ns = [Nx]
    
    else: # choose where to place the waterline
        left = True
        Lcl = Lb - Lr # contraction length to the left of the waterline
        Lcr = Ly - Lb # contraction length to the right of the waterline
        ratios = []   # list of ratios of left to right resolutions
        
        for i in range(Nx//2 - 1): # loop over options
            Nl = 2*(i+1)  # no. of columns to the left of the waterline
            Nr = Nx - Nl  # no. of columns to the right of the waterline
            dycl = Lcl/Nl # y grid resolution to the left of the waterline
            dycr = Lcr/Nr # y grid resolution to the right of the waterline
            ratios.append(abs(dycl/dycr-1))
        
        choice = ratios.index(min(ratios)) # chosen waterline index
        Nl = 2*(choice+1)
        Nr = Nx - Nl
        Ns = [Nl, Nr]
        dycl = Lcl/Nl
        dycr = Lcr/Nr
        print(f'The ratio of resolutions for the chosen waterline is {dycl/dycr}')
    
    for N in Ns:
        if Lb is None:
            dyc = Lc/Nx # grid resolution in y direction
            dycl = None
            dycr = None
        elif left is True:
            dyc = dycl
        else:
            dyc = dycr
        
        for col in range(N): # loop over columns
            if left is True:
                nk = Nr + N - col # no. of elements in column
            else:
                nk = N - col
            nnL = nk + 1 + col%2 # no. of nodes on left-hand line
            nnR = nk + (col+1)%2 # no. of nodes on right-hand line
            
            if left is False:
                yL = Lb + col*dyc # y pos of LH line
            else:
                yL = Lr + col*dyc
            yR = yL + dyc        # y pos of RH line
            
            xL = Lx*(Ly-yL)/Lc   # length of LH line
            xLt = (Lx-xL)/2      # x pos of top of LH line
            xLb = Lx - (Lx-xL)/2 # x pos of bottom of LH line
            dxl = xL/nk          # grid resolution along LH line
            
            xR = Lx*(Ly-yR)/Lc   # length of RH line
            xRt = (Lx-xR)/2      # x pos of top of RH line
            xRb = Lx - (Lx-xR)/2 # x pos of bottom of RH line
            if nk != 1:
                dxr = xR/(nk-1)  # grid resolution along RH line
            
            for j in range(nk): # loop over elements, grouped by shape
                if j < nk/2 - 1: # top parallelograms
                    x1 = xLt + j*dxl
                    x2 = x1 + dxl
                    x3 = xRt + (j+1)*dxr
                    x4 = x3 - dxr
                    
                    y1 = yL
                    y2 = yL
                    y3 = yR
                    y4 = yR
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + nnL
                    n4 = n3 - 1
                
                elif j > nk/2: # bottom parallelograms
                    x1 = xLb - (nk-j)*dxl
                    x2 = x1 + dxl
                    x3 = xRb - (nk-j-1)*dxr
                    x4 = x3 - dxr
                    
                    y1 = yL
                    y2 = yL
                    y3 = yR
                    y4 = yR
                    
                    n1 = Nn + nnL - (nk-j+1)
                    n2 = n1 + 1
                    n3 = n2 + nnR
                    n4 = n3 - 1
                
                elif col%2 == 1: # kites
                    x1 = (Lx-dxl)/2
                    x2 = Lx/2
                    x3 = (Lx+dxl)/2
                    x4 = Lx/2
                    
                    y1 = yL
                    y2 = yL - dyc/2
                    y3 = yL
                    y4 = yR
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + 1
                    n4 = n3 + nnR
                
                elif j == nk/2 - 1: # top scalene quadrilaterals
                    x1 = Lx/2 - dxl
                    x2 = Lx/2
                    x3 = Lx/2
                    x4 = (Lx-dxr)/2
                    
                    y1 = yL
                    y2 = yL
                    y3 = yL + dyc/2
                    y4 = yR
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + nnL
                    n4 = n3 - 1
                
                else: # bottom scalene quadrilaterals
                    x1 = Lx/2
                    x2 = Lx/2 + dxl
                    x3 = (Lx+dxr)/2
                    x4 = Lx/2
                    
                    y1 = yL
                    y2 = yL
                    y3 = yR
                    y4 = yL + dyc/2
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + nnR
                    n4 = n3 - 1
                
                mesh_data.write(f'{x1:<10f} {y1:<10f} {x2:<10f} {y2:<10f} {x3:<10f} {y3:<10f} {x4:<10f} {y4:<10f} {n1:{D}d} {n2:{D}d} {n3:{D}d} {n4:{D}d}\n')
            
            Nn += nnL # no. of first node on LH line of next column
        
        if left is True:
            left = False
    
    ### Finish
    Nn += 1 # Nn currently gives no. of last node; +1 to get total no. of nodes
    
    mesh_data.close()
    
    return (Lr, Nn, dycl, dycr)

#%% get_mesh
def get_mesh(Lx, Ly, Nx, Ny, d=None, Lc=None, Lr=None, Lb=None):
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
    
    ### generate DMPlex - need lists of node indices of each cell and coordinates of each node
    # see https://www.firedrakeproject.org/_modules/firedrake/mesh.html and search 'def _from_cell_list'
    
    # write grid file, load parameters, read data and delete file
    Lr, Nn, dycl, dycr = create_grid(Lx, Ly, Nx, Ny, d, Lc, Lr, Lb)
    data = np.loadtxt(f'mesh_Lx={Lx}_Ly={Ly}_Lr={Lr:.3f}_Nx={Nx}_Ny={Ny}_Lb={Lb}.txt')
    os.remove(f'mesh_Lx={Lx}_Ly={Ly}_Lr={Lr:.3f}_Nx={Nx}_Ny={Ny}_Lb={Lb}.txt')
    
    # get node indices of each cell
    cells = data[:, [8, 9, 10, 11]].astype(int) # data is read as float so convert to int
    # cells[i, j] returns (global) index of jth (local) node of row i+1 in file
    
    # get coordinates of each node
    xys = data[:, :8].reshape(-1, 4, 2)
    # xys[i, j, k] returns x coord (if k=0) or y coord (if k=1) of jth (local) node of row i+1 in file
    coords = np.empty((Nn, 2), dtype=float)
    coords[cells] = xys
    # coords[i, j] returns x coord (if j=0) or y coord (if j=1) of ith (global) node
    
    plex = mesh.plex_from_cell_list(2, cells, coords, COMM_WORLD)
    
    ### Apply boundary IDs to DMPlex - see https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/singleindex.html
    coord_sec = plex.getCoordinateSection() # Section for vecGetClosure()
    coordinates = plex.getCoordinates()     # Vec for vecGetClosure()
    
    # mark wavemaker
    plex.markBoundaryFaces('boundary_faces')                             # set label for edges along boundary
    boundary_faces = plex.getStratumIS('boundary_faces', 1).getIndices() # array of 3*Nx + 2*Ny boundary faces
    tol = Lr/Ny/2                                                        # tolerance for round-off error on wavemaker
    for face in boundary_faces:                                          # loop over faces on boundary
        face_coords = plex.vecGetClosure(coord_sec, coordinates, face)   # x0,y0,x1,y1 of nodes forming face
        if face_coords[1] < tol and face_coords[3] < tol:                # if face is on y=0 boundary
            plex.setLabelValue(cython.dmcommon.FACE_SETS_LABEL, face, 0) # 0 = wavemaker
    
    # mark waterline
    if Lb is not None:
        ltol = dycl/2                                                        # tolerance for round-off error left of waterline
        rtol = dycr/4                                                        # tolerance for round-off error right of waterline
        for node in range(*plex.getDepthStratum(0)):                         # loop over all (interior and exterior) nodes
            node_coords = plex.vecGetClosure(coord_sec, coordinates, node)   # x,y of node
            if node_coords[1] < Lb - ltol:                                   # if node is left of y=Lb line
                plex.setLabelValue(cython.dmcommon.FACE_SETS_LABEL, node, 1) # 1 = left of waterline
            elif Lb - ltol < node_coords[1] < Lb + rtol:                     # if node is on y=Lb line
                plex.setLabelValue(cython.dmcommon.FACE_SETS_LABEL, node, 2) # 2 = waterline
            else:                                                            # if node is right of y=Lb line
                plex.setLabelValue(cython.dmcommon.FACE_SETS_LABEL, node, 3) # 3 = right of waterline
    
    return mesh.Mesh(plex), Lr
