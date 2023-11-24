__author__ = "Bianca Bodo"
__project_name__ = "Dissertation"

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, image_channels, mesh_feature_dim, output_dim):
        super(Generator, self).__init__()
        # Image branch
        self.fc_image = nn.Linear(150528 * 224, 512)
        self.bn_image = nn.BatchNorm1d(512)
        self.activation_image = nn.LeakyReLU(0.2)

        # Mesh branch
        self.fc_mesh = nn.Linear(mesh_feature_dim, 512)
        self.bn_mesh = nn.BatchNorm1d(512)
        self.activation_mesh = nn.LeakyReLU(0.2)

        # Combined branch
        self.fc_combined = nn.Linear(512 * 2, 512)
        self.bn_combined = nn.BatchNorm1d(512)
        self.activation_combined = nn.LeakyReLU(0.2)


        self.fc_output = nn.Linear(512, output_dim)
        self.output_activation = nn.Tanh()

        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_image.weight)
        nn.init.xavier_uniform_(self.fc_mesh.weight)
        nn.init.xavier_uniform_(self.fc_combined.weight)
        nn.init.xavier_uniform_(self.fc_output.weight)

    def forward(self, noise, image, mesh_features):
        # Reshape input tensors
        # image = image.view(image.size(0), -1)
        # image = image.to(self.fc_image.weight.dtype)  # Flatten image
        # mesh_features = mesh_features.view(mesh_features.size(0), -1)
        # mesh_features = mesh_features.to(self.fc_mesh.weight.dtype)  # Flatten mesh features
        # print("Mesh shape before fc_image:", mesh_features.shape)
        # print("fc_mesh weight shape:", self.fc_mesh.weight.shape)
        # image_embedding = torch.relu(self.fc_image(image))
        # mesh_embedding = torch.relu(self.fc_mesh(mesh_features))
        # print("image_embedding shape:", image_embedding.shape)
        # print("mesh_embedding shape:", mesh_embedding.shape)
        # combined = torch.cat((image_embedding, mesh_embedding), dim=1)
        # print("combined shape:", combined.shape)
        # combined = torch.relu(self.fc_combined(combined))
        # output = torch.tanh(self.fc_output(combined))
        # return output

        image_embedding = self.activation_image(self.bn_image(self.fc_image(image)))
        mesh_embedding = self.activation_mesh(self.bn_mesh(self.fc_mesh(mesh_features)))
        combined = torch.cat((image_embedding, mesh_embedding), dim=1)
        combined = self.activation_combined(self.bn_combined(self.fc_combined(combined)))
        output = self.output_activation(self.fc_output(combined))
        return output
