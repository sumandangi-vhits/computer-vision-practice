import torch
import torch.nn as nn

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(32 * 56 * 56, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


device = torch.device("cpu")

model = SimpleCNN().to(device)

image = torch.randn(1, 3, 224, 224)

output = model(image)

print("Input shape:", image.shape)
print("Output shape:", output.shape)

print("Model output:", output)
print("Device:", device)