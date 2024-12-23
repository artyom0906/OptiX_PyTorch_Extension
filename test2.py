# test2.py
import math

import torch
import optix_renderer
import pygame
import numpy as np
import time
import trimesh
from PIL import Image

from python.animation.animation import Translate, Animator, Rotate
from python.loaders.Model import Model
from python.loaders.OBJLoader import OBJLoader
from python.controllers.gamepad_controller import GamepadController
# Initialize pygame
pygame.init()

# Create a window
width, height = 1920, 1080
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("OptiX Renderer Output")

def load_custom_geometry():
    vertices = np.array([
        [-1.0, -1.0, -1.0],  # Vertex 0
        [1.0, -1.0, -1.0],   # Vertex 1
        [1.0, 1.0, -1.0],    # Vertex 2
        [-1.0, 1.0, -1.0],   # Vertex 3
        [-1.0, -1.0, 1.0],   # Vertex 4
        [1.0, -1.0, 1.0],    # Vertex 5
        [1.0, 1.0, 1.0],     # Vertex 6
        [-1.0, 1.0, 1.0],    # Vertex 7
    ], dtype=np.float32)

    indices = np.array([
        [0, 1, 2], [0, 2, 3],  # Front face
        [4, 5, 6], [4, 6, 7],  # Back face
        [0, 4, 7], [0, 7, 3],  # Left face
        [1, 5, 6], [1, 6, 2],  # Right face
        [3, 2, 6], [3, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Bottom face
    ], dtype=np.uint32)

    return vertices, indices

def rotate_cube_in_place(vertices_tensor, angle_x, angle_y, angle_z):
    # Compute rotation matrices
    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)

    rot_x = torch.tensor([[1, 0, 0],
                          [0, cos_x, -sin_x],
                          [0, sin_x, cos_x]], device='cuda')

    rot_y = torch.tensor([[cos_y, 0, sin_y],
                          [0, 1, 0],
                          [-sin_y, 0, cos_y]], device='cuda')

    rot_z = torch.tensor([[cos_z, -sin_z, 0],
                          [sin_z, cos_z, 0],
                          [0, 0, 1]], device='cuda')

    # Combine the rotation matrices
    rotation_matrix = rot_z @ rot_y @ rot_x

    # Apply the rotation matrix in-place
    vertices_tensor[:] = torch.matmul(vertices_tensor, rotation_matrix.T)
def apply_transformation_in_place(transform_matrix, rotation_angles, scale, translation):
    """
    Apply rotation (3 axes), scaling, and translation to a 3x4 transformation matrix in-place.

    Parameters:
        transform_matrix (torch.Tensor): Existing 3x4 transformation matrix (modified in-place).
        rotation_angles (tuple): Rotation angles (rx, ry, rz) in radians for X, Y, and Z axes.
        scale (float): Scaling factor.
        translation (torch.Tensor): Translation vector (length 3).

    Returns:
        None: Modifies the `transform_matrix` in-place.
    """
    if transform_matrix.shape != (3, 4):
        raise ValueError("Transform matrix must be 3x4.")
    if translation.shape != (3,):
        raise ValueError("Translation vector must be of size 3.")

    rx, ry, rz = rotation_angles

    # Rotation matrix for X-axis
    cos_rx = math.cos(rx)
    sin_rx = math.sin(rx)
    rotation_x = torch.tensor([
        [1, 0,       0      ],
        [0, cos_rx, -sin_rx],
        [0, sin_rx,  cos_rx]
    ], dtype=torch.float32, device=transform_matrix.device)

    # Rotation matrix for Y-axis
    cos_ry = math.cos(ry)
    sin_ry = math.sin(ry)
    rotation_y = torch.tensor([
        [ cos_ry, 0, sin_ry],
        [ 0,      1, 0     ],
        [-sin_ry, 0, cos_ry]
    ], dtype=torch.float32, device=transform_matrix.device)

    # Rotation matrix for Z-axis
    cos_rz = math.cos(rz)
    sin_rz = math.sin(rz)
    rotation_z = torch.tensor([
        [cos_rz, -sin_rz, 0],
        [sin_rz,  cos_rz, 0],
        [0,       0,      1]
    ], dtype=torch.float32, device=transform_matrix.device)

    # Combined rotation matrix (Z * Y * X)
    rotation_matrix = torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))

    # Apply scaling to the rotation matrix
    rotation_scaling_matrix = rotation_matrix * scale

    # Update the rotation part (R₀₀ to R₂₂) in-place
    transform_matrix[:, :3] = torch.matmul(rotation_scaling_matrix, transform_matrix[:, :3])

    # Update the translation part (T₀ to T₂) in-place
    transform_matrix[:, 3] += translation

def load_obj_with_trimesh(file_path, device=None):
    """
    Load a .obj file using trimesh and convert its vertices and faces into PyTorch tensors on the specified device.

    Parameters:
        file_path (str): Path to the .obj file.
        device (torch.device, optional): The device to load tensors onto. If None, automatically selects GPU if available.

    Returns:
        vertices_tensor (torch.FloatTensor): Tensor of shape (num_vertices, 3).
        faces_tensor (torch.LongTensor): Tensor of shape (num_faces, 3).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mesh = trimesh.load(file_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file does not contain a single mesh.")
    visuals = mesh.visual
    print(vars(visuals.material), visuals.material, visuals.uv)
    tmp_texture_tensor = None
    if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
        image = visuals.material.image.convert('RGBA')

        # Convert image data to uint8 numpy array
        image_data = np.array(image).astype(np.uint8)  # Ensure the data type is uint8

        # Convert to a tensor and ensure it's on the GPU
        tmp_texture_tensor = torch.tensor(image_data, dtype=torch.uint8, device=device)
    else:
        print("No image path found in the material.")

# Convert mesh data to PyTorch tensors and move to the specified device
    vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.uint32, device=device)
    uv_tensor = torch.tensor(visuals.uv, dtype=torch.float32, device=device) if visuals.uv is not None else None

    return vertices_tensor, faces_tensor, uv_tensor, tmp_texture_tensor


# Convert to PyTorch tensors and move to GPU
#vertices_tensor = torch.tensor(vertices, device='cuda', dtype=torch.float32)
#indices_tensor = torch.tensor(indices, device='cuda', dtype=torch.uint32)
def loadImage(path):
    image = Image.open(path)
    image = image.convert('RGBA')
    # Convert image data to uint8 numpy array
    image_data = np.array(image).astype(np.uint8)  # Ensure the data type is uint8
    # Convert to a tensor and ensure it's on the GPU
    image = torch.tensor(image_data, dtype=torch.uint8, device=device)
    #print(image)
    texture_pot = optix_renderer.TextureObject(image)
    return texture_pot

device = torch.device("cuda")  # Use GPU if available
# Initialize camera parameters
camera_position = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # Eye position
camera_lookat = np.array([0.0, 0.0, 0.0], dtype=np.float32)    # Look-at point
camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)        # Up vector

move_speed = 0.2  # Adjust movement speed as needed

# Create a CUDA tensor for the output image
output_tensor = torch.empty((height, width, 4), dtype=torch.uint8, device='cuda')

# Instantiate the Renderer
renderer = optix_renderer.Renderer(width, height)

print("renderer done")
texture_pot = loadImage(r"models/elevator3/textures/normals.png")
texture_pot = loadImage(r"models/elevator3/textures/normals.png")
# obj_loader = OBJLoader(r"models/backpack/backpack.obj", r"models/backpack/diffuse.jpg", normal_file=r"models/backpack/normal.png")
#obj_loader = OBJLoader(r"models/cyborg/cyborg.obj", r"models/cyborg/cyborg_diffuse.png", normal_file=r"models/cyborg/cyborg_normal.png")
#obj_loader = OBJLoader(r"models/damaged-wall/SM_Wall_Damaged_2x1_A.obj", r"models/damaged-wall/textures/T_Wall_Damaged_2x1_A_BC.png",
#                       normal_file=r"models/damaged-wall/textures/T_Wall_Damaged_2x1_A_N.png")
#obj_loader = OBJLoader(r"models/elevator4/3.obj", r"models/elevator4/textures/color.png", normal_file=r"models/elevator4/textures/normal.png")
#obj_loader = OBJLoader(r"models/gun/gun.obj", r"models/gun/textures/famas-f1-atb-skin_baseColor.jpeg", normal_file=r"models/gun/textures/famas-f1-atb-skin_normal.png")
obj_loader = OBJLoader(r"models/bt/test_bb_model.obj", r"models/bt/test_bb_texture.png", normal_file=r"models/bt/test_bb_normal_normals.png")

# #model = obj_loader.add(renderer)
# # # Load custom geometry
# vertices, indices, uv, image = load_obj_with_trimesh(r"models/glass-door/glass_door_framless.obj")
# print("Vertices Tensor on device:")
# #print(vertices)
# print("Indices Tensor on device:")
# #print(indices)
# print("UV Tensor on device:")
# #print(uv)
# print("Indices Tensor on device:")

texture_pot = loadImage(r"models/teapot/default.png")
texture_pot1 = loadImage(r"models/test/testTexture.png")

# #geometry = renderer.createVertexGeometry(vertices, indices, uv, texture_pot)
teapot_model = obj_loader.load()#Model(vertices, indices, uv, texture_pot)
print("add geometry py")
#tea_ring_id = renderer.addGeometryInstance(teapot_model.create_geometry(renderer))
teapot_model_instance0 = teapot_model.add(renderer)
teapot_model_instance1 = teapot_model.add(renderer, color=(0, 1, 0), texture=texture_pot)
teapot_model_instance2 = teapot_model.add(renderer, color=(0, 0, 1), texture=texture_pot1)

#obj_loader1 = OBJLoader(r"models/elevator3/untitled.obj", r"models/elevator3/textures/iron_bars.png_baseColor.png", normal_file=r"models/backpack/normal.png")

#elevator3_model = obj_loader1.load()

#elevator3_model_instance = elevator3_model.add(renderer)
# # Example usage
# rootPath = r"models/test"
# gltf_loader = GLTFLoader(rootPath, 'scene.gltf')
# vertices_tensors = gltf_loader.get_vertices()
# indices_tensors = gltf_loader.get_indices()
# tex_coords_tensors = gltf_loader.get_tex_coords()
# textures = gltf_loader.get_textures()
# # Printing tensor information
# print("Number of Meshes Loaded:", len(vertices_tensors))
# texture_map = {}
# for idx, vertices_tensor in enumerate(vertices_tensors):
#     print(f"Mesh {idx}: Vertices Tensor:", vertices_tensor.shape)
#     print(f"Mesh {idx}: Indices Tensor:", indices_tensors[idx].shape)
#     if tex_coords_tensors[idx] is not None:
#         print(f"Mesh {idx}: Texture Coordinates Tensor:", tex_coords_tensors[idx].shape)
#     texture_tensor = gltf_loader.get_texture_for_mesh(idx)
#     print(image, texture_tensor, image.shape, texture_tensor.shape)
#     indices_tensor = indices_tensors[idx]
#     if texture_tensor is not None:
#         print(f"Mesh {idx}: Associated Texture Shape:", texture_tensor.shape)
#     else:
#         continue
#     if not texture_tensor.is_contiguous():
#         texture_tensor = texture_tensor.contiguous()
#     if not vertices_tensor.is_contiguous():
#         vertices_tensor = vertices_tensor.contiguous()
#     if not indices_tensor.is_contiguous():
#         indices_tensor = indices_tensor.contiguous()
#
#     tex_coords_tensor = tex_coords_tensors[idx]
#     if not tex_coords_tensor.is_contiguous():
#         tex_coords_tensor = tex_coords_tensor.contiguous()
#
#     texture = optix_renderer.TextureObject(texture_tensor)
#     #print(texture)
#     geometry = renderer.createVertexGeometry(vertices_tensor, indices_tensor, tex_coords_tensor, texture)
#     ring_id = renderer.addGeometryInstance(geometry)
#     texture_map[idx] = ring_id
#     #print("ringid:", ring_id)

print("geometry done")
renderer.buidIAS()
print("ias done")

#transform_matrix = renderer.getTransformForInstance(teapot_model_instance.geometry_id-1)
#translation = torch.tensor([0.0, 10.0, 0.0], dtype=torch.float32, device=device)  # Translation vector
#apply_transformation_in_place(transform_matrix, (180-45, 0, 0), 1/20, translation)
teapot_model_instance0.set_transform([ 0.0, 10.0,  0.0], [np.radians(180), np.radians(0), np.radians(0)], [1/5, 1/5, 1/5])
teapot_model_instance1.set_transform([10.0, 10.0,  0.0], [np.radians(180), np.radians(0), np.radians(0)], [1/5, 1/5, 1/5])
teapot_model_instance2.set_transform([ 0.0, 10.0, 10.0], [np.radians(180), np.radians(0), np.radians(0)], [1/5, 1/5, 1/5])
#elevator3_model_instance.set_transform([ 0.0, 0.0, 0.0], [np.radians(180), np.radians(0), np.radians(0)], [1/2, 1/2, 1/2])
#model.set_transform(translation=[0.0, 10.0, 0.0], rotation=[180-45, 0, 0], scale=[1, 1, 1])

# for idx in texture_map.keys():
#     transform_matrix = renderer.getTransformForInstance(texture_map[idx]-1)
#     transform = torch.tensor(gltf_loader.get_translation_matrix_by_mesh_id(idx), dtype=torch.float32, device='cuda')
#     transform_matrix[:3, :4] = transform[:3, :4]

transform = Translate([0, 0, 0], [0, -10, 0]) * Rotate([180, 0, 0], [180, 90, 0])
animator = Animator([teapot_model_instance0], transform, 3000)

print("load transform done")

controller = GamepadController()
running = True
clock = pygame.time.Clock()

# Variables for frame time measurement
frame_times = [0]
num_frames = 0

fps = 0
# Initialize rotation angles
yaw = 0
camera_position = torch.tensor([0, -5, -30], device='cuda', dtype=torch.float32)
while running:
    # Record the start time
    start_time = time.perf_counter()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        camera_position[2] -= move_speed  # Move forward
        camera_lookat[2] -= move_speed    # Move lookat point forward
    if keys[pygame.K_s]:
        camera_position[2] += move_speed  # Move backward
        camera_lookat[2] += move_speed    # Move lookat point backward
    if keys[pygame.K_a]:
        camera_position[0] -= move_speed  # Move left
        camera_lookat[0] -= move_speed    # Move lookat point left
    if keys[pygame.K_d]:
        camera_position[0] += move_speed  # Move right
        camera_lookat[0] += move_speed    # Move lookat point right
    if keys[pygame.K_q]:
        camera_position[1] -= move_speed  # Move down
        camera_lookat[1] -= move_speed    # Move lookat point down
    if keys[pygame.K_e]:
        camera_position[1] += move_speed  # Move up
        camera_lookat[1] += move_speed    # Move lookat point up

    # Camera parameters: concatenate into a list
    camera_params = camera_position.tolist() + camera_lookat.tolist() + camera_up.tolist()

    # Fill the first three columns with the identity matrix
    #transform_tensor[:, :3] = torch.eye(3, dtype=transform_tensor.dtype, device=transform_tensor.device)


    controller.update()

    # Get movement inputs
    move_x, move_y, move_z = controller.get_movement()
    rotate, pitch_input = controller.get_rotation()
    #adjusted_zoom = self.controller.get_zoom()

    #rotate = 0.1

    # Adjust camera rotation based on controller input
    yaw += rotate * controller.rotation_speed

    # Update camera's view direction based on yaw
    view_direction = torch.tensor([
        np.sin(yaw),
        0.0,
        -np.cos(yaw)
    ], device='cuda', dtype=torch.float32)

    # Compute the forward and right vectors relative to the camera's orientation
    forward = view_direction / torch.norm(view_direction)
    right = torch.cross(torch.tensor([0.0, 1.0, 0.0], device='cuda'), forward, dim=0)
    right = right / torch.norm(right)
    up = torch.tensor([0.0, 1.0, 0.0], device='cuda', dtype=torch.float32)

    # Compute movement direction based on controller input
    move_direction = (move_x * right +
                      move_y * forward +
                      move_z * up) * controller.movement_speed
    camera_position += move_direction

    # Adjust camera position along the view direction for zoom
    #self.camera_position += view_direction

    lookat = camera_position + view_direction

    # Detach tensors from CUDA and convert to CPU for further processing
    eye = camera_position.detach().cpu().numpy()
    lookat = lookat.detach().cpu().numpy()
    up_np = up.detach().cpu().numpy()

    # Combine into a single array if needed
    camera_params = np.concatenate((eye, lookat, up_np))

    animator.update(frame_times[-1]*1000)
# Call the render function
    renderer.render(output_tensor, camera_params)

    # Copy tensor to CPU and convert to numpy array
    output_image = output_tensor.cpu().numpy()

    # Convert to 8-bit per channel
    #output_image = (np.clip(output_image, 0.0, 1.0) * 255).astype(np.uint8)
    output_image = np.transpose(output_image, (1, 0, 2))
    # Convert to pygame surface
    surface = pygame.surfarray.make_surface(output_image[:, :, :3])

    # Display the surface
    window.blit(surface, (0, 0))

    # Calculate and display frame time
    end_time = time.perf_counter()
    frame_time = end_time - start_time
    frame_times.append(frame_time)
    num_frames += 1

    # Calculate average frame time every 60 frames
    if num_frames % 60 == 0:
        avg_frame_time = sum(frame_times[-60:]) / 60.0
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        print(f"Average frame time: {avg_frame_time * 1000:.2f} ms, FPS: {fps:.2f}")

    # Optionally, display the FPS on the window title
    pygame.display.set_caption(f"OptiX Renderer Output - FPS: {fps:.2f}")

    pygame.display.flip()
    clock.tick(0)  # Remove frame cap to measure actual performance

# Clean up
pygame.quit()

