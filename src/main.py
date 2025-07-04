import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import *

# python3 main.py

torch.manual_seed(0)

x_samples = torch.linspace(-1, 1, 100).unsqueeze(1)
y_samples = x_samples

num_units = 100
lr = 0.01
epochs = 20000

model = SimpleModel(num_units)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss()

loss_history = []
for epoch in range(epochs):
    y_pred = model(x_samples)
    loss = loss_function(y_pred, y_samples)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.6f}")

with torch.no_grad():
    fitted_y = model(x_samples)

a_values = model.a.detach().numpy()
b_values = model.b.detach().numpy()
c_values = model.c.detach().numpy()
print(np.linalg.norm(a_values))
print(np.linalg.norm(b_values))
print(np.linalg.norm(c_values))

plt.figure(figsize=(8, 6))
plt.plot(x_samples.numpy(), y_samples.numpy(), label="Target", color='blue')
plt.plot(x_samples.numpy(), fitted_y.numpy(), label="Fitted", color='red', linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("plot.pdf")
