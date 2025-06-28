import torch

x=torch.randn(3,requires_grad=True)
print(x)
y=x+10
print(y)
z=y*y*2
z=z.sum()
#check it for scalar values in the arguments ex z=y*y*2 is nt taken scalar
#another way to tackle this prblm is take a vector for jacobian product and pass it to the .backward() as argument(ex.[0.1,1.0,0.001]) with remving of .sum/.mean func
print(z)
z.backward()
print(x.grad)

#how to prevent gradient trekking: 
#1.requires_grad=false
#2.x.detach()
#3.with torch.no_grad

#for not summing up the iteration of the model for next,use tensor.grad.zero_()
#for optimization use torch.optim.SGD(weights,lr=0.01)