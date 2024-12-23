import pygame


class SelectionWheel:
    def __init__(self, items, cell_sizes, screen_width, screen_height, bar_height_ratio=0.22):
        self.items = items
        self.cell_sizes = cell_sizes
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.bar_height = screen_height * bar_height_ratio
        self.selected_index = 0
        self.font = pygame.font.Font(None, 36)

    def get_selected_item(self):
        return self.items[((len(self.cell_sizes))//2 - self.selected_index) % len(self.items)]

    def draw_text(self, surface, text, position, size):
        text_font = pygame.font.Font(None, size * 10)
        text_surface = text_font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=position)
        surface.blit(text_surface, text_rect)

    def draw_cell(self, surface, text, position, size, is_selected=False):
        cell_size = size * 20
        cell_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)

        if is_selected:
            pygame.draw.rect(cell_surface, (255, 255, 255, 150), (0, 0, cell_size, cell_size), border_radius=10)

        pygame.draw.rect(cell_surface, (200, 200, 200, 100), (0, 0, cell_size, cell_size), border_radius=10)
        surface.blit(cell_surface, (position[0] - cell_size // 2, position[1] - cell_size // 2))

        self.draw_text(surface, text, position, size)

    def render(self, surface):
        bar_y = self.screen_height - self.bar_height
        #pygame.draw.rect(surface, (0, 0, 0), (0, bar_y, self.screen_width, self.bar_height))

        base_y = self.screen_height - self.bar_height / 2
        self.draw_text(surface, "LB", (self.screen_width * 0.36, base_y - 15), 3)
        self.draw_text(surface, "RB", (self.screen_width * 0.64, base_y - 15), 3)

        cell_spacing = 15
        total_width = sum(size * 20 for size in self.cell_sizes) + (len(self.cell_sizes) - 1) * cell_spacing
        start_x = (self.screen_width - total_width) // 2

        for i, size in enumerate(self.cell_sizes):
            cell_width = size * 20
            pos_x = start_x + sum(self.cell_sizes[:i]) * 20 + i * cell_spacing + cell_width // 2
            text = self.items[(i - self.selected_index) % len(self.items)]
            self.draw_cell(surface, text, (pos_x, base_y - cell_width // 2), size, is_selected=(i == len(self.cell_sizes) // 2))

    def update_selection(self, increment):
        self.selected_index = (self.selected_index + increment) % len(self.items)