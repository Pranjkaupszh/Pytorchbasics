import torch
# Linear regression
# f = w * x 
#  (f = 2 * x)
X = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
Y = torch.tensor([4, 8, 12, 16], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
# model output
def forward(x):
    return w * x
# loss = Mean sq err(MSE)
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.4f}')
# Training
learning_rate = 0.01
for epoch in range(100):
    # predict = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # calculate gradients = backward pass
    l.backward()
    # update weights
    #w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero the gradients after updating
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.6f}')
print(f'Prediction after training: f(5) = {forward(5).item():.4f}')