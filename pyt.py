import torch

x=torch.rand(3,3)
y=torch.rand(3,3)
y.sub_(x);
print(y)
#basic func add,sub 
#to perform inline operations in pytorch(ex.(x+-*)y), funcs have trailing underscore ex add_,sub_,etc..
x.view(-1,8) #view is used for resizing the tensor