# Paco Carnazza 2024
# All errors reserved

import torch
import numpy as np
from torch.autograd import grad
from torch.linalg import det,inv
from math import pi, sqrt
from torch.autograd.functional import hessian

def product_gaussian(mu1,sigma1,mu2,sigma2):
    '''
    method to compute the product of two gaussians with 
    respective center and covarince mu1, mu2 and sigma1, sigma 2
    '''
    if len(mu1) != len(mu2) or len(mu1) != len(sigma2):
            raise Exception("Length of mu and sigma must be equal to dim")
    return torch.sqrt( (2 * pi) ** len(mu1) * det(sigma1+sigma2)) * torch.exp(0.5*(mu1-mu2) @ inv(sigma1+sigma2) @ (mu1-mu2)) 

def product_gaussian_(x):
    '''
    method to compute the product of two gaussians with 
    respective center and covarince mu1, mu2 and sigma1, sigma 2
    '''
    dim = int((len(x)-2) * 0.5)
    mu1 = x[0]
    sigma1 = x[1:(dim+1)]
    mu2 = x[dim+1]
    sigma2 = x[(dim+2):]
    if len(mu1) != len(mu2) or len(mu1) != len(sigma2):
            raise Exception("Length of mu and sigma must be equal to dim")
    return torch.sqrt( (2 * pi) ** len(mu1) * det(sigma1+sigma2)) * torch.exp(0.5*(mu1-mu2) @ inv(sigma1+sigma2) @ (mu1-mu2)) 

if __name__ == '__main__':
    sigma = torch.tensor([[2.,3.],[3.,2.]])
    sigma1 = torch.tensor(sigma.T @ sigma,requires_grad=True) 
    mu1 = torch.tensor([[2.,3.]],requires_grad=True)
    sigma2 = torch.tensor(sigma.T @ sigma,requires_grad=True) 
    mu2 = torch.tensor([[3.,3.]],requires_grad=True)
    x = torch.cat((mu1,sigma1,mu2,sigma2))
    pg = product_gaussian_(x)
    print(pg)
    print(grad(pg,x,create_graph=True)[0])
    
    print(hessian(product_gaussian_,x,create_graph=True))
    
    
