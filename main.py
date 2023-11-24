__author__ = "Bianca Bodo"
__project_name__ = "Dissertation"

import torch
import mayavi.mlab as mlab
import PyQt5
import numpy as np
from loading_functions.loading import load_image, load_mesh
from GAN.load_GAN import Generator

MODEL_PATH = "/Users/biancabodo/Downloads/BAT1_SETA_HOUSE2.obj"
IMAGE_PATH = ("/Users/biancabodo/Desktop/Diss/datasets/Arch styles/architectural-styles-dataset/"+
              "Gothic architecture/02_0004.jpg")

noise_dim = 100
image_channels = 1
mesh_feature_dim = 3
output_dim = 3

def generate_3d_mesh(generator, random_noise, input_image, input_mesh_features):
    generated_output = generator(random_noise, input_image, input_mesh_features)
    # Convert the generated output to a NumPy array (adjust this based on your actual output format)
    generated_mesh_np = generated_output.detach().numpy()
    return generated_mesh_np

input_image = load_image(IMAGE_PATH)
# input_image = input_image.view(1, -1)  # Flatten image
input_mesh_features = load_mesh(MODEL_PATH)
mesh_features = torch.tensor(input_mesh_features).view(1, -1)


random_noise = torch.randn(1, noise_dim)
generator = Generator(noise_dim, image_channels, mesh_features.size(1), output_dim)

generated_mesh_np = generate_3d_mesh(generator, random_noise, input_image, mesh_features)

# Reshape the generated_mesh_np array based on your output_dim
# For example, if output_dim is 3 and each row represents a vertex, you might reshape it as follows:
vertices = generated_mesh_np[:, :3]  # Assuming the first three columns are vertex coordinates
triangles = generated_mesh_np[:, 3:]  # Assuming the rest of the columns are triangle indices

print(generated_mesh_np)
# Plot the 3D mesh using mayavi
if vertices.size == 0 or triangles.size == 0:
    print("Generated mesh data is empty.")
    # Handle the case of empty data appropriately
else:
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles, scalars=None, colormap='Blues')
    mlab.show()