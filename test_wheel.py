import pygame

from python.models.SelectionWheel import SelectionWheel

# Initialize pygame
pygame.init()

# Screen dimensions
width, height = 1920, 1080
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Selection Wheel")

# Initialize selection wheel
items = ["A", "B", "C", "D", "E"]
cell_sizes = [2, 4, 7, 4, 2]
selection_wheel = SelectionWheel(items, cell_sizes, width, height)

running = True
clock = pygame.time.Clock()

# Initialize joystick
pygame.joystick.init()
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.JOYBUTTONDOWN and joystick:
            if joystick.get_button(4):  # Left bumper
                selection_wheel.update_selection(-1)
            elif joystick.get_button(5):  # Right bumper
                selection_wheel.update_selection(1)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  # Simulate left bumper
                selection_wheel.update_selection(-1)
            elif event.key == pygame.K_RIGHT:  # Simulate right bumper
                selection_wheel.update_selection(1)

    window.fill((0, 0, 0))
    selection_wheel.render(window)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
