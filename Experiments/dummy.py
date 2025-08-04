# test_unext_integration.py

import torch
from nets.UNext import UNext  # Make sure path is correct

# Dummy config
n_channels = 3
n_labels = 1
img_size = 224
batch_size = 2

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNext(n_channels=n_channels, n_classes=n_labels).to(device)
model.eval()
x = torch.randn(batch_size, n_channels, img_size, img_size).to(device)





# Forward pass
with torch.no_grad():
    y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
