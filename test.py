import math
import time

import numpy as np
import torch
import optix_renderer
import pygame
from PIL import Image

from python.animation.animation import Translate, Animator, Rotate, Scale
from python.controllers.GamepadController import GamepadController
from python.elevator.Door import Door, DoorBuilder
from python.elevator.Elevator import ElevatorBuilder
from python.elevator.Hallway import HallwayBuilder
from python.loaders.ModelInstance import apply_rotation
from python.loaders.OBJLoader import OBJLoader
from python.models.Camera import Camera
from python.models.Player import Player
from python.models.SelectionWheel import SelectionWheel

#from python.models.text import draw_text

# Initialize pygame
pygame.init()

from python.models.text import draw_text

# Create a window
width, height = 1920, 1080
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("OptiX Renderer Output")

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

obj_loader_hall = OBJLoader(r"models/scene/Hallway/Hallway.obj", r"models/scene/Hallway/HallwayTexture.png", normal_file=r"models/scene/Hallway/HallwayNormals.png")
obj_loader_hall_door = OBJLoader(r"models/scene/HallwayDoor/HallwayDoor.obj", r"models/scene/HallwayDoor/HallwayDoorTexture.png", normal_file=r"models/scene/HallwayDoor/HallwayDoorNormals.png")
obj_loader_lift_door_r = OBJLoader(r"models/scene/LiftDoor/LiftDoorRight.obj", r"models/scene/LiftDoor/LiftDoorTexture.png", normal_file=r"models/scene/LiftDoor/LiftDoorNormals.png")
obj_loader_lift_door_l = OBJLoader(r"models/scene/LiftDoor/LiftDoorLeft.obj", r"models/scene/LiftDoor/LiftDoorTexture.png", normal_file=r"models/scene/LiftDoor/LiftDoorNormals.png")
obj_loader_lift_tb = OBJLoader(r"models/scene/LiftTB/LiftTB.obj", r"models/scene/LiftTB/LiftTBTexture.png", normal_file=r"models/scene/LiftTB/LiftTBNormals.png",
                               emission_texture_file=r"models/scene/LiftTB/LiftTBEmission.png")
obj_loader_elevator = OBJLoader(r"models/scene/Lift/Lift.obj", r"models/scene/Lift/LiftTexture.jpg",
                                normal_file=r"models/scene/Lift/LiftNormals.png", flip_textures=(False, False, False, False))
#obj_loader = OBJLoader(r"models/bt/test_bb_model.obj", r"models/bt/test_bb_texture.png", normal_file=r"models/bt/test_bb_normal_normals.png")
obj_loader_gun = OBJLoader(r"models/gun/gun.obj", r"models/gun/textures/famas-f1-atb-skin_baseColor.jpeg", normal_file=r"models/gun/textures/famas-f1-atb-skin_normal.png")
obj_loader = OBJLoader(r"models/simple_studio_light/light.obj", r"models/simple_studio_light/textures/LMP0001_Textures_baseColor.png",
                       normal_file=r"models/simple_studio_light/textures/LMP0001_Textures_normal.png", emission_texture_file=r"models/simple_studio_light/textures/LMP0001_Textures_emissive.png",
                       metallic_roughness_file=r"models/simple_studio_light/textures/LMP0001_Textures_metallicRoughness.png", flip_textures=(True, True, True, True))


texture_pot = loadImage(r"models/teapot/default.png")
texture_pot1 = loadImage(r"models/test/testTexture.png")

# #geometry = renderer.createVertexGeometry(vertices, indices, uv, texture_pot)
model = obj_loader.load()#Model(vertices, indices, uv, texture_pot)
gun_model = obj_loader_gun.load()


hallway_model = obj_loader_hall.load()
hall_door_model = obj_loader_hall_door.load()
elevator_model = obj_loader_elevator.load()
lift_door_r_model = obj_loader_lift_door_r.load()
lift_door_l_model = obj_loader_lift_door_l.load()
lift_tb_model = obj_loader_lift_tb.load()



print("add geometry py")
instances = []
#tea_ring_id = renderer.addGeometryInstance(teapot_model.create_geometry(renderer))
instances.append((model.add(renderer), ([ 0.0, 10.0,  0.0], [np.radians(180), np.radians(0), np.radians(0)], [1, 1, 1])))
instances.append((model.add(renderer, color=(0, 1, 0), texture=texture_pot), ([2.0, 10.0,  1.0], [np.radians(180), np.radians(90), np.radians(0)], [1, 1, 1])))
instances.append((model.add(renderer, color=(0, 0, 1), texture=texture_pot1),([ 0.0, 10.0, 2.0], [np.radians(180), np.radians(180), np.radians(0)], [1, 1, 1])))

#obj_loader1 = OBJLoader(r"models/elevator3/untitled.obj", r"models/elevator3/textures/iron_bars.png_baseColor.png", normal_file=r"models/backpack/normal.png")

#elevator3_model = obj_loader1.load()


renderer.buidIAS()

for i in instances:
    i[0].set_transform(*i[1])

renderer.buidIAS()

for i in instances:
    i[0].reset_transform_matrix()
    i[0].set_transform(*i[1])

#renderer.buidIAS()
#instance.set_transform([10, 10, 10], [np.radians(180), np.radians(0), np.radians(0)], [1, 1, 1])

transform = Translate([0, 10, 0], [0, 10, 0]) * Rotate([180, 0, 0], [180, 0, 0])
#animator = Animator([teapot_model_instance0], transform, 3000)

print("load transform done")

controller = GamepadController()  # Or KeyboardController()
camera = Camera([0, -5, -30], [0, 0, 0], [0, 1, 0])
player = Player(controller, camera)

hall_door_builder = DoorBuilder(renderer, hall_door_model, Translate([0, 0, 0], [0, 32, 0]), Translate([0, 0, 0], [0, -32, 0]))
lift_door_r_builder = DoorBuilder(renderer, lift_door_r_model, Translate([0, 0, 0], [0, 0, 8]), Translate([0, 0, 0], [0, 0, -8]))
lift_door_l_builder = DoorBuilder(renderer, lift_door_l_model, Translate([0, 0, 0], [0, 0, -8]), Translate([0, 0, 0], [0, 0, 8]))
elevator_builder = ElevatorBuilder(renderer, elevator_model, lift_tb_model, [lift_door_r_builder, lift_door_l_builder])
hallway_builder = HallwayBuilder(renderer, hallway_model, [hall_door_builder])
items = {"A": model,
         "B": gun_model,
         "C": hallway_model,
         "D": None,
         "E": None,
         "F": None,
         "G": None,
         "H": None,
         "I": None,
         "J": None,
         "K": None,
         "L": None}


cell_sizes = [2, 4, 7, 4, 2]
selection_wheel = SelectionWheel(list(items.keys()), cell_sizes, width, height)
animations = []

running = True
clock = pygame.time.Clock()

# Variables for frame time measurement
frame_times = [0]
num_frames = 0

fps = 0
# Initialize rotation angles
yaw = 0
camera_position = torch.tensor([0, -5, -30], device='cuda', dtype=torch.float32)
interaction_text = ""
pressed = False

hall_doors = []
elevator_doors = []
elevators = []
hallways = []
while running:
    scene_updated = False
    # Record the start time
    start_time = time.perf_counter()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.JOYAXISMOTION and controller.joysticks[0]:
            scene_updated = True
        elif event.type == pygame.JOYBUTTONDOWN and controller.joysticks[0]:
            scene_updated = True
            if controller.joysticks[0].get_button(5):  # Left bumper
                selection_wheel.update_selection(-1)
            elif controller.joysticks[0].get_button(4):  # Right bumper
                selection_wheel.update_selection(1)
            elif controller.joysticks[0].get_button(0) and not pressed:
                interaction_text = f"selected {selection_wheel.get_selected_item()}"
                m = items[selection_wheel.get_selected_item()]
                scale = 1
                if selection_wheel.get_selected_item() == "I":
                    for hallway in hallways:
                        hallway.open_doors()
                elif selection_wheel.get_selected_item() == "J":
                    for hallway in hallways:
                        hallway.close_doors()

                if selection_wheel.get_selected_item() == "K":
                    for elevator in elevators:
                        elevator.open_doors()
                if selection_wheel.get_selected_item() == "L":
                    for elevator in elevators:
                        elevator.close_doors()

                if selection_wheel.get_selected_item() in ["C", "D"]:
                    scale = 0.1

                if selection_wheel.get_selected_item() == "D":
                    hall_doors.append(hall_door_builder.build([float(camera.position[0]), float(camera.position[1]), float(camera.position[2])],
                                                                                 [np.radians(180), np.radians(0), np.radians(0)],
                                                                                 [scale, scale, scale]))
                    renderer.buidIAS()
                    for i in instances:
                        i[0].reset_transform_matrix()
                        i[0].set_transform(*i[1])
                    for door in elevator_doors:
                        door.reset_position()
                    for door in hall_doors:
                        door.reset_position()
                    for elevator in elevators:
                        elevator.reset_position()
                if selection_wheel.get_selected_item() in ["G", "H"]:
                    if selection_wheel.get_selected_item() == "G":
                        builder = lift_door_l_builder
                    else:
                        builder = lift_door_r_builder
                    elevator_doors.append(builder.build([float(camera.position[0]), float(camera.position[1]), float(camera.position[2])],
                                                                  [np.radians(180), np.radians(0), np.radians(0)],
                                                                  [scale, scale, scale]))
                    renderer.buidIAS()
                    for i in instances:
                        i[0].reset_transform_matrix()
                        i[0].set_transform(*i[1])
                    for door in elevator_doors:
                        door.reset_position()
                    for door in hall_doors:
                        door.reset_position()
                    for elevator in elevators:
                        elevator.reset_position()
                    for hallway in hallways:
                        hallway.reset_position()

                if selection_wheel.get_selected_item() == "E":
                    elevators.append(elevator_builder.build([float(camera.position[0]), float(camera.position[1]), float(camera.position[2])],
                                                            [np.radians(180), np.radians(0), np.radians(0)],
                                                            [0.1, 0.1, 0.1]))
                    renderer.buidIAS()
                    for i in instances:
                        i[0].reset_transform_matrix()
                        i[0].set_transform(*i[1])
                    for door in elevator_doors:
                        door.reset_position()
                    for door in hall_doors:
                        door.reset_position()
                    for elevator in elevators:
                        elevator.reset_position()
                    for hallway in hallways:
                        hallway.reset_position()

                if selection_wheel.get_selected_item() == "C":
                    hallways.append(hallway_builder.build([float(camera.position[0]), float(camera.position[1]), float(camera.position[2])],
                                                            [np.radians(180), np.radians(0), np.radians(0)],
                                                            [0.1, 0.1, 0.1]))
                    renderer.buidIAS()
                    for i in instances:
                        i[0].reset_transform_matrix()
                        i[0].set_transform(*i[1])
                    for door in elevator_doors:
                        door.reset_position()
                    for door in hall_doors:
                        door.reset_position()
                    for elevator in elevators:
                        print([float(camera.position[0]), float(camera.position[1]), float(camera.position[2])])
                        elevator.add_floor([float(camera.position[0])*10, -float(camera.position[1]+5)*10, -float(camera.position[2]+30)*10])
                        elevator.reset_position()
                    for hallway in hallways:
                        hallway.reset_position()

                if selection_wheel.get_selected_item() == "B":
                    for elevator in elevators:
                        elevator.next_floor()

                if selection_wheel.get_selected_item() in ["A"]:
                    instances.append((m.add(renderer, emission=(15, 15, 15)),
                                      ([float(camera.position[0]), float(camera.position[1]), float(camera.position[2])],
                                       [np.radians(180), -camera.yaw+math.pi, np.radians(0)],
                                       [scale, scale, scale])
                                      ))
                    renderer.buidIAS()
                    for i in instances:
                        i[0].reset_transform_matrix()
                        i[0].set_transform(*i[1])
                    for door in elevator_doors:
                        door.reset_position()
                    for door in hall_doors:
                        door.reset_position()
                    for elevator in elevators:
                        elevator.reset_position()
                    for hallway in hallways:
                        hallway.reset_position()
                pressed = True

            elif controller.joysticks[0].get_button(1):
                interaction_text = ""
                pressed = False
    player.update(frame_times[-1])
    for animaton in animations:
        animaton.update(frame_times[-1]*1000)
        scene_updated = True
    for door in hall_doors:
        if door.update(frame_times[-1]*1000):
            scene_updated = True
    for door in elevator_doors:
        if door.update(frame_times[-1]*1000):
            scene_updated = True
    for elevator in elevators:
        if elevator.update(frame_times[-1]*1000):
            scene_updated = True
    for hallway in hallways:
        if hallway.update(frame_times[-1]*1000):
            scene_updated = True
    animations = [a for a in animations if not a.is_complete]
    # Camera parameters: concatenate into a list
    camera_params = camera.position.tolist() + camera.lookat.tolist() + camera.up.tolist()

    #animator.update(frame_times[-1]*1000)
# Call the render function
    renderer.render(output_tensor, camera_params, scene_updated)

    # Copy tensor to CPU and convert to numpy array
    output_image = output_tensor.cpu().numpy()

    # Convert to 8-bit per channel
    #output_image = (np.clip(output_image, 0.0, 1.0) * 255).astype(np.uint8)
    output_image = np.transpose(output_image, (1, 0, 2))
    # Convert to pygame surface
    surface = pygame.surfarray.make_surface(output_image[:, :, :3])

    # Display the surface
    window.blit(surface, (0, 0))
    selection_wheel.render(window)


    # Render text below the center of the screen
    if interaction_text:
        screen_center = (width // 2, height // 2)
        text_position = (screen_center[0], screen_center[1] + 50)  # Slightly below center
        draw_text(window, interaction_text, text_position, (255, 0, 255))

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

