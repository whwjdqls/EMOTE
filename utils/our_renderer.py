import torch
import os

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
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)

from models.flame_models import flame
import numpy as np
import json
from torchvision import transforms
import time

#import trimesh
#from PIL import Image
#import imageio.v2 as imageio   

def get_texture_from_template(template_path, device):
    """get texture from template

    Args:
        template_path (str): path to template
        device (torch.device): device
    return:
        tex (TexturesUV): TexturesUV from template obj file
        Note that there is only one tex. use tex.extend(BS) to get a list of tex
    """
    # get textures from head_template.obj
    verts, faces, aux = load_obj(template_path)
    verts_uvs = aux.verts_uvs.to(device)    # ( V, 2)
    faces_uvs = faces.textures_idx.to(device) # (F, 3)
    tex_maps = aux.texture_images
    image = list(tex_maps.values())[0].to(device)[None]
    tex = TexturesUV(
            verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
        )
    return tex


def render_flame(config, vertices, faces, textures, device):
    """render flame model
    (note) this is a function that makes BS*T meshes...is it too memory consuming?
    Args:
        config (dict): config file
        vertices (torch.tensor): verticies (BS*T, 5023, 3)
        faces (torch.tensor): template FLAME Faces (BS*T,9976,3)
        textures (TexturesUV): TexturesUV from template obj file length BS*T
        device (torch.device): device
    return:
        images (torch.tensor): rendered images (BS*T, 256, 256, 4)
    """
    BS = vertices.shape[0]
    # get mesh from vertices
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    # get camera
    T = torch.tensor([[0., 0.01, 0.2]]).repeat(BS,1)
    R = torch.tensor([[[-1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0., -1.]]]).repeat(BS,1,1)
    znear = torch.tensor(0.01).to(device)
    cameras = FoVPerspectiveCameras(znear=znear, device=device, R=R, T=T)
    # get lights
    lights = PointLights(device=device, location=[[0., 1.,2.]])
    # get rasterization settings
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # get renderer
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
    # render
    images = renderer(meshes)
    return images

def render_flame_lip(config, vertices, faces, textures, device):
    """render flame model
    (note) this is a function that makes BS*T meshes...is it too memory consuming?
    Args:
        config (dict): config file
        vertices (torch.tensor): verticies (BS*T, 5023, 3)
        faces (torch.tensor): template FLAME Faces (BS*T,9976,3)
        textures (TexturesUV): TexturesUV from template obj file length BS*T
        device (torch.device): device
    return:
        images (torch.tensor): rendered images (BS*T, 88, 88, 4)
    """
    BS = vertices.shape[0]
    # get mesh from vertices
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    # get camera
    T = torch.tensor([[0., 0.0475, 0.09]]).repeat(BS,1)
    R = torch.tensor([[[-1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0., -1.]]]).repeat(BS,1,1)
    znear = torch.tensor(0.01).to(device)
    cameras = FoVPerspectiveCameras(znear=znear, device=device, R=R, T=T)
    # get lights
    lights = PointLights(device=device, location=[[0., 1.,2.]])
    # get rasterization settings
    raster_settings = RasterizationSettings(
        image_size=88,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # get renderer
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
    # render
    images = renderer(meshes)
    return images



def to_lip_reading_image(images):
    """convert to images that are input to lip reading model
    (NOTE) this function is for lip reading PER FRAME
    Args:
        images (torch.tensor): rendered images (BS*T,3, 256, 256) or (BS*T, 3, 88, 88)
    return:
        images (torch.tensor): rendered images (BS*T, 1, 1, 88, 88)
    """

    if images.shape[-1] == 256:
        images = transforms.functional.crop(images, 140, 85, 88, 88) # (BS*T, 3, 88, 88)
    grayscaled = transforms.functional.rgb_to_grayscale(images) # (BS*T, 1, 88, 88)
    return grayscaled.unsqueeze(1) # (BS*T, 1, 1, 88, 88)

# (12-14) pyrender not working for NOW..
def pyrender_flame(vertices, faces, image_save_dir):
    """render flame model with pyrender
    Args:
        vertices (torch.tensor): verticies (T, 5023, 3)
        faces (torch.tensor): template FLAME Faces (9976,3)
    
    """
    for i in range(vertices.shape[0]):
        vertices_ = vertices[i].detach().cpu().numpy()
        vertex_colors = np.ones([vertices_.shape[0], 4]) * [0.5, 0.5, 0.5, 1.0]
        tri_mesh = trimesh.Trimesh(vertices_, faces, vertex_colors=vertex_colors,
                               face_colors=[0.7, 0.7, 0.7, 1.0], process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        # create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
        camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.35],  # Adjust Z position
                                [0.0, 0.0, 0.0, 1.0]])
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene = pyrender.Scene(nodes=[camera_node],
                            bg_color = [0.0, 0.0, 0.0, 1.0])
        scene.add(mesh)

        # create light
        light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=0.5)
        scene.add(light)
        # create renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=1400, viewport_height=1080)
        # render
        color, _ = renderer.render(scene)
        image = Image.fromarray(color)
        # save
        imageio.imwrite(f'{image_save_dir}/{i+1:03d}.png', color)

    return
