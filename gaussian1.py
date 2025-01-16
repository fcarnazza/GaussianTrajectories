# Paco Carnazza 2024
# All errors reserved

import torch
from torch import nn
from math import pi, sqrt
from torch.linalg import inv, det
from torch.autograd import grad

class Gaussian(nn.Module):
    '''
    Class to define a multivariate Gaussian normalized to const
    mu: Tensor, vector of centers.
    sigma: Tensor, covariance matrix.
    dim: int, multivariate dimension.
    '''
    def __init__(self,mu, sigma,dim,const=1):
        super().__init__()
        if len(mu) != dim or len(sigma) != dim:
            raise Exception("Length of mu and sigma must be equal to dim")
        self.mu = mu
        self.sigma = sigma
        det = torch.linalg.det(sigma)
        self.N = det * sqrt(2*pi)**dim   
        self.const = const
        self.inv_sigma = inv(sigma)
        self.dim = dim
    def forward(self,x):
        return self.const / self.N * torch.exp(- 0.5 * (x-self.mu).T @ self.inv_sigma @ (x-self.mu))

def prod(g1,g2):
    '''
    method to compute the product of two Gaussians 
    return an object of type Gaussian
    Input:
    g1: Gaussian
    g2: Gaussian
    Output:
    Gaussian
    '''
    sigma3 = inv(g1.inv_sigma + g2.inv_sigma)
    mu3 = sigma3 @(g1.inv_sigma @ g1.mu + g2.inv_sigma @ g2.mu)
    C = g1.sigma + g2.sigma
    z3 = torch.sqrt(det( C )) * torch.exp(0.5 * (g1.mu-g2.mu) @ (inv(C)) @ (g1.mu-g2.mu)) *((2*pi)**dim)
    n3 = g1.const * g2.const / z3
    return Gaussian(mu3, sigma3,const = n3)

class BilateralLaplaceGaussian(nn.Module):
    '''
    class to compute the gaussian resulting from the
    bilateral Laplace transform of an input gaussian.
    https://en.wikipedia.org/wiki/Two-sided_Laplace_transform
    '''
    def __init__(self,g_input):
        super().__init__()
        self.g_input = g_input
        self.dim = g_input.dim
    def forward(self,j):
        return self.g_input.const * torch.exp(-0.5 * j.T @ self.g_input.sigma @ j + j @ self.g_input.mu)


def gaussian_moment(indxs,gaus_in):
    '''
    method to compute the multivariate moment of a 
    Gaussian.
    Input:
    indxs: list of integers for the moment one is
    computing. E.g. if you want to compute the third moment of variable x0:
    <x0^3>, use indxs = [0,0,0]
    '''
    gaus = BilateralLaplaceGaussian(gaus_in)
    j0 = torch.zeros(gaus.dim,requires_grad=True).float()
    gaus0 = gaus(j0)
    grad_out = grad(gaus0,j0,create_graph=True)[0]
    for i in range(len(indxs)-1):
        grad_out = grad(grad_out[indxs[i]],j0,create_graph=True)[0]
    return grad_out[indxs[-1]]



if __name__ =='__main__':
    sigma = torch.tensor([[2.,3.],[3.,2.]])
    sigma = sigma.T @ sigma 
    mu = torch.tensor([2.,3.])
    dim =2

    g0 = Gaussian(mu,sigma,dim)
    print(g0(mu))
    print(gaussian_moment([1,1,1,0,0,0],g0))

