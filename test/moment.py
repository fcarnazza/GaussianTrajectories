import torch
from torch import autograd


def gaussian(x):
    return torch.exp(0.5 * x * 2 * x + x*2)

x = torch.tensor(0., requires_grad = True)
g1 = gaussian(x)
first_derivative = autograd.grad(g1, x, create_graph=True)[0]
# We now have dgaussian/dx
print(first_derivative)
second_derivative = autograd.grad(first_derivative, x)[0]
# This computes d/dx(dgaussian/dx) = d2gaussian/dx2
print(second_derivative)


a = torch.tensor([[2.,3.],[3.,2.]])
a = a.T @ a
b = torch.tensor([2.,3.])
x = torch.tensor([0.,0.], requires_grad = True)
def multi_gaussian(x):
    return torch.exp(0.5 * x.T @ a @ x + x.T@b)

g1 = multi_gaussian(x)
first_derivative = autograd.grad(g1, x, create_graph=True)[0]
# We now have dgaussian/dx

print(first_derivative)
 
second_derivative = autograd.grad(first_derivative[1], x,create_graph = True)[0]
print(second_derivative)
second_derivative = autograd.grad(second_derivative[1], x,create_graph = True)[0]
# This computes d/dx(dloss/dx) = d2loss/dx2
print(second_derivative)




