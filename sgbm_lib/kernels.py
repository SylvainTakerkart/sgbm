"""
This is a simple file containing useful kernels.
On the contrary to other implementations I made
they are based on the array class of numpy 
(not matrix)
"""

from numpy import *

class gaussian:
    
    def __init__(self, sigma=1.0):
        """ sets the parameter of the gaussian kernel

        sigma is the traditional variance parameter
        gamma is just 1/\sigma^2
        """
        self.sigma=sigma


    def evaluate(self,u,v):
        """
        Just evaluates the kernel. Matrix computations
        makes it quicker than loops
        u, v: the sets of vectors on which the kernel must be
        evaluated. Each row corresponds to an example

        returns: the kernels between the sets of vectors
        """
        
        u_dim = u.shape[0]
        v_dim = v.shape[0]
        
        u_1_norms  = sum(u**2, axis=1)
        sq_u       = tile(u_1_norms, (v_dim, 1)).T
        
        v_1_norms = sum(asarray(v)**2, axis=1)
        sq_v      = tile(v_1_norms, (u_dim, 1))
        
       
        dot_products = dot(u, v.T)
        
        kernel = exp((2*dot_products - sq_u - sq_v) / (2*self.sigma**2))
        
        return kernel

    
    def gram(self, u):
        """
        Computes the Gram matrix of u
        u: the matrix from which the gram matrix should be computed
        each row corresponds to an example

        returns the Gram matrix
        """
        u_dim = u.shape[0]
        
        u_norms = sum(u**2, axis=1),
        sq_u = tile(u_norms, (u_dim,1))
        
        dot_products = dot(u, u.T)
         
        kernel = exp((2*dot_products - sq_u - sq_u.T) / (2*self.sigma**2))

        return kernel
    

class polynomial:
    """
    Computes a polynomial kernel
    
    As usual, it is computed as k(u,v)=(a*u\cdot v+b)^p,
    where u and v might be matrices of line vectors, i.e. each example
    is provided as a row vector
    """

    def __init__(self, a=1, b=0, p=1.0):
        """
        sets the parameter of the polynomial kernel
        """
        self.a = a
        self.b = b
        self.p = p


    def evaluate(self,u,v):
        """
        Just evaluates the kernel. Matrix computations
        makes it quicker than loops
        """
        dot_products = dot(u, v.T)

        kernel = (self.a * dot_products + self.b) ** self.p

        return kernel


    def gram(self, u):
        """
        Computes the Gram matrix of u
        """
        dot_products = dot(u, u.T)
    
        kernel = (self.a * asarray(dot_products) + self.b) ** self.p

        return kernel
    

""" Some tests """
if __name__ == "__main__":
    
    u = random.uniform(0,1,(10,5))
    v = random.uniform(0,1,(5,5))

    print u.shape
    print v.shape

    k = gaussian()

    print 'noyau entre u et v'
    print k.evaluate(u,v)

    print 'matrice de gram'
    print k.gram(u)

    kpoly =polynomial()

    print 'noyau entre u et v'
    print kpoly.evaluate(u,v)

    print 'matrice de gram'
    print kpoly.gram(u)
