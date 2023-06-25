import numpy as np

class Integrator:
    """
    Description: Base class for defining different integrators
    """
    def __init__(self, domain, num, dtype='float32'):
        self.domain = domain 
        self.dtype = dtype
        self.set_nodes(num)


    def set_nodes(self, num):
        self.nodes = np.zeros(num).astype(self.dtype)
        self.weights = np.ones(num).astype(self.dtype)
        
    def compute(self, f):
        return (f(self.nodes)*self.weights).sum()
    
    def quad(self, evals):
        return (evals*self.weights).sum()
    

class Trapezoidal(Integrator):
    """
    Description: Class for defining trapezoidal quadrature
    """
    def __init__(self, domain, num, dtype='float32'):
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
        self.nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        self.h = (self.domain[1] - self.domain[0]) / num
        self.weights = np.ones(len(self.nodes)) 
        self.weights[0] = 0.5 
        self.weights[-1] = 0.5
        self.weights *= self.h
    

class Simpson_1_3(Integrator):
    """
    Description: Class for defining Simpson 1/3 quadrature
    """
    def __init__(self, domain, num, dtype='float32'):
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
        """
        num: an even number
        """
        self.nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        h = (self.domain[1] - self.domain[0]) / num
        self.weights = np.ones(len(self.nodes)) 
        for i in range(1,num):
            if i%2 == 0:
                self.weights[i] = 2.
            else:
                self.weights[i] = 4.
        self.weights *= (h/3.)



class Simpson_3_8(Integrator):
    """
    Description: Class for defining Simpson 3/8 quadrature
    """
    def __init__(self, domain, num, dtype='float32'):
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
        """
        num: a multiple of 3
        """
        self.nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        h = (self.domain[1] - self.domain[0]) / num
        self.weights = np.ones(len(self.nodes)) 
        for i in range(1,num):
            if i%3 == 0:
                self.weights[i] = 2.
            else:
                self.weights[i] = 3.
        self.weights *= (3.*h/8.)


class Gauss_Legendre(Integrator):
    """
    Description: Class for defining Gauss-Legendre quadrature
    """
    def __init__(self, domain, num, d, dtype='float32'):
        self.d = d
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
        self.x, self.w = np.polynomial.legendre.leggauss(self.d)
        self.x, self.w = self.x.astype(self.dtype), self.w.astype(self.dtype)
        pre_nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        h = (pre_nodes[1] - pre_nodes[0]) / 2
        self.nodes = []
        self.weights = np.array(list(self.w) * num) * h
        for i in range(num):
            a = pre_nodes[i]
            b = pre_nodes[i+1]
            c, d = (b-a)/2., (b+a)/2.
            self.nodes += list(c*self.x+d) 
        self.nodes = np.array(self.nodes)


class Gauss_Legendre_2D():
    """
    Description: Class for defining Gauss-Legendre quadrature
    """
    def __init__(self, domain, num, d, dtype='float32'):
        self.d = d
        self.domain = domain 
        self.dtype = dtype
        self.nodes = [[], []]
        self.weights = [[], []]
        self.set_nodes(0, num)
        self.set_nodes(1, num)
        self.x, self.y = np.meshgrid(self.nodes[0], self.nodes[1])
        self.wx, self.wy = np.meshgrid(self.weights[0], self.weights[1])
        self.x = self.x.reshape(-1, 1)
        self.y = self.y.reshape(-1, 1)
        self.wx = self.wx.reshape(-1, 1)
        self.wy = self.wy.reshape(-1, 1)
        self.w = self.wx * self.wy

    def set_nodes(self, i, num):
        self.x, self.w = np.polynomial.legendre.leggauss(self.d)
        self.x, self.w = self.x.astype(self.dtype), self.w.astype(self.dtype)
        pre_nodes = np.linspace(self.domain[0][i], self.domain[1][i], num=num+1, endpoint=True, dtype=self.dtype)
        h = (pre_nodes[1] - pre_nodes[0]) / 2
        
        self.weights[i] = np.array(list(self.w) * num) * h
        for j in range(num):
            a = pre_nodes[j]
            b = pre_nodes[j+1]
            c, d = (b-a)/2., (b+a)/2.
            self.nodes[i] += list(c*self.x+d) 
        self.nodes[i] = np.array(self.nodes[i])

    


class Trapezoidal_2D():
    """
    Description: Class for defining Gauss-Legendre quadrature
    """
    def __init__(self, domain, num, dtype='float32'):
        self.domain = domain 
        self.dtype = dtype
        self.nodes = [[], []]
        self.weights = [[], []]
        self.set_nodes(0, num)
        self.set_nodes(1, num)
        self.x, self.y = np.meshgrid(self.nodes[0], self.nodes[1])
        self.wx, self.wy = np.meshgrid(self.weights[0], self.weights[1])
        self.x = self.x.reshape(-1, 1)
        self.y = self.y.reshape(-1, 1)
        self.wx = self.wx.reshape(-1, 1)
        self.wy = self.wy.reshape(-1, 1)
        self.w = self.wx * self.wy

    def set_nodes(self, i, num):
        self.nodes[i] = np.linspace(self.domain[0][i], self.domain[1][i], num=num+1, endpoint=True, dtype=self.dtype)
        self.h = (self.domain[1][i] - self.domain[0][i]) / num
        self.weights[i] = np.ones(len(self.nodes[i])) 
        self.weights[i][0] = 0.5 
        self.weights[i][-1] = 0.5
        self.weights[i] *= self.h


# Code for testing
# import scipy.integrate as integrate
# domain = [0.001, 20.]
# num = 100
# f = lambda x: x*np.sin(x) 
# t = Trapezoidal(domain, num*100)
# t_= t.compute(f)
# s1 = Simpson_1_3(domain, num*100)
# s1_ = s1.compute(f)
# s2 = Simpson_3_8(domain, num*100)
# s2_ = s2.compute(f)
# g = Gauss_Legendre(domain, int(num), 100)
# g_ = g.compute(f)
# s_ = integrate.quad(f, *domain)[0]
# r = s_#np.log(domain[1]/domain[0])
# def u(a, r):
#     return np.abs(np.log10(np.abs(a-r)))
# print('Trapezoidal ---> {}, {}'.format(t_, u(t_, r)))
# print('Simpson 1/3 ---> {}, {}'.format(s1_, u(s1_, r)))
# print('Simpson 3/8 ---> {}, {}'.format(s2_, u(s2_, r)))
# print('Gauss-Legendre ---> {}, {}'.format(g_, u(g_, r)))
# print('Scipy ---> {}, {}'.format(s_, u(s_, r)))
# print('True ---> {}'.format(r))
    