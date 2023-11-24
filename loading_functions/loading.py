__author__ = "Bianca Bodo"
__project_name__ = "Dissertation"

from PIL import Image
import open3d as o3d
import torchvision.transforms as transforms
import numpy as np


def load_image(model_path):
    # Open the image file
    image = Image.open(model_path).convert("RGB")  # Ensure the image is in RGB mode

    # Preprocess the image (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust the size as needed
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Adjust normalization if needed
    ])

    # Apply the transformations to the image
    processed_image = transform(image).unsqueeze(0)  # Add batch dimension

    return processed_image

def load_mesh(model_path):
    mesh = o3d.io.read_triangle_mesh(model_path)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    input_data = np.concatenate((vertices, faces), axis=0)

    return input_data
