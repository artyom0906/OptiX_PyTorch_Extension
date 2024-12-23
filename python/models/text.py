import pygame

font = pygame.font.Font(None, 36)  # Use default font, size 36

# Function to draw text
def draw_text(screen, text, position, color=(255, 255, 255)):
    # Render the text
    text_surface = font.render(text, True, color)
    # Get the text rectangle for positioning
    text_rect = text_surface.get_rect(center=position)
    # Blit the text surface to the screen
    screen.blit(text_surface, text_rect)
