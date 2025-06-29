import torch

# Inputs and target
x = torch.tensor(4.0)
y = torch.tensor(2.0)

# Initialize weight
w = torch.tensor(1.0, requires_grad=True)

# Learning rate
lr = 0.01

# Training loop
for epoch in range(10):
    # Forward pass: compute predicted y
    yh = w * x
    
    # Compute loss(sq error)
    loss = (yh - y) ** 2
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}, w = {w.item():.4f}")

    # Backward pass
    loss.backward()

    # Update weight using gradient descent
    with torch.no_grad():
        w -= lr * w.grad

    # Zero the gradients after updating
    w.grad.zero_()
