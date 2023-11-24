__author__ = "Bianca Bodo"
__project_name__ = "Dissertation"

import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Assuming mesh data is in the range [-1, 1]
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model_path = "/Users/biancabodo/Downloads/BAT1_SETA_HOUSE2.obj"
mesh = o3d.io.read_triangle_mesh(model_path)

vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
input_data = np.concatenate((vertices, faces), axis=0)

num_vertices = vertices.shape[0]
num_faces = faces.shape[0]
start_input_size = 3 * (num_vertices + num_faces)

# Define input size based on your representation of mesh data
input_size = start_input_size
output_size = start_input_size

# Instantiate Generator and Discriminator
generator = Generator(input_size, output_size)
discriminator = Discriminator(input_size)

# Print the architectures to understand their structures
print(generator)
print(discriminator)
print('puf')