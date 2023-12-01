import torch
import os
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
device = torch.device("cuda:0")
path = "/workspace/audio2mesh/emoca/gdl_apps/EMOCA/demos/image_output/EMOCA_v2_lr_mse_20/0000_0000_0000/EMOCA_v2_lr_mse_20/inputs00/mesh_coarse.obj"
# mesh = load_obj(path, path=True)
mesh = load_objs_as_meshes([path], device=device)
# print(type(mesh))
# print('len mesh: ', len(mesh))

# data_tuple = mesh
# verts_tensor = data_tuple[0]
# faces_tensor = data_tuple[1].verts_idx
# normals_tensor = data_tuple[1].normals_idx
# textures_tensor = data_tuple[1].textures_idx
# materials_tensor = data_tuple[1].materials_idx
# verts_uvs_tensor = data_tuple[2].verts_uvs

# # Accessing additional information (e.g., material colors)
# material_colors_dict = data_tuple[2].material_colors
# texture_image = data_tuple[2].texture_images
# # print name and shape of all tensors
# print("verts_tensor: ", verts_tensor.shape)
# print("faces_tensor: ", faces_tensor.shape)
# print("normals_tensor: ", normals_tensor.shape)
# print("textures_tensor: ", textures_tensor.shape)
# print("materials_tensor: ", materials_tensor.shape)
# print("verts_uvs_tensor: ", verts_uvs_tensor.shape)
# print("material_colors_dict: ", material_colors_dict)
# print("texture_image: ", texture_image['FaceTexture'].shape)

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
# R, T = look_at_view_transform(2.7, 0, 180) 
R, T = look_at_view_transform(0.8, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)



images = renderer(mesh)
print(images)
print(images.shape)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
# save plt image
plt.savefig('test_0.8.png')



R, T = look_at_view_transform(0.6, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

images = renderer(mesh)
print(images)
print(images.shape)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
# save plt image
plt.savefig('test_0.6.png')