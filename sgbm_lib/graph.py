class graph:
    def __init__(self, L, A, V, X, N, run, label):
        # label of each node of the graph (the implementation of the graph kernels only works if this is +1 and or -1)
        self.L = L

        # adjacancy matrix
        self.A = A

        # functional attributes of each node
        self.V = V

        # geometrical attributes (coordinates) of each node
        self.X = X

        # number of the experimental session
        self.run  = run

        # number of the class this sample belongs to
        self.label = label

        # number of original voxels in each node/parcel
        self.N = N

class pitsgraph:
    def __init__(self, A, X, D, S=None, T=None, other_coords=None):
        # adjacancy matrix
        self.A = A

        # coordinates of each pit
        self.X = X

        # attributes of the pits: depth
        self.D = D

        # attributes of the basins: area (surface)
        self.S = S

        # attributes of the basins: mean thickness
        self.T = T

        # sotre other coordinates just in case (here, the spherical coordinates, rho & theta)
        self.other_coords = other_coords
        
        

        
