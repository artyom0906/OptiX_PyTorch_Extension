# pygame_monitor.py

import pygame
import threading
import time
import numpy as np
import torch
import subprocess
import traceback
import random

# --- GPU Monitoring ---
def get_gpu_info():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=2
        )
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip(): continue
            values = [v.strip() for v in line.split(',')]
            if len(values) >= 7:
                gpu_stats.append({
                    'id': int(values[0]), 'name': values[1],
                    'gpu_util': float(values[2]), 'memory_util': float(values[3]),
                    'memory_used': float(values[4]), 'memory_total': float(values[5]),
                    'temperature': float(values[6])
                })
        return gpu_stats
    except Exception:
        return []


class PygameMonitor:
    def __init__(self, vr_render_width_per_eye, vr_render_height_per_eye,
                 monitor_window_width=1920, monitor_window_height=1080):

        self.vr_render_width_original = vr_render_width_per_eye
        self.vr_render_height_original = vr_render_height_per_eye

        self.pygame_target_window_width = monitor_window_width
        self.pygame_target_window_height = monitor_window_height

        self.is_running = False
        self.thread = None
        self.data_lock = threading.Lock()
        self.latest_monitor_data = {
            "left_eye_cpu_tensor": None, "right_eye_cpu_tensor": None,
            "left_render_ms": 0.0, "left_optix_copy_ms": 0.0, "left_texture_copy_ms": 0.0,
            "left_gpu_to_cpu_ms": 0.0, "left_cpu_to_gpu_ms": 0.0,
            "right_render_ms": 0.0, "right_optix_copy_ms": 0.0, "right_texture_copy_ms": 0.0,
            "right_gpu_to_cpu_ms": 0.0, "right_cpu_to_gpu_ms": 0.0,
            "total_to_cpu_ms": 0.0, "total_from_cpu_ms": 0.0,
            "vr_fps": 0.0, "system_gpu_stats": [], "main_loop_time_ms": 0.0
        }

        self.screen = None
        self.clock = None

        self.font_small_size = 22
        self.font_medium_size = 26
        self.font_large_size = 30

        self.font_small = None
        self.font_medium = None
        self.font_large = None

        self.frames_area_height = int(self.pygame_target_window_height * 0.75)
        self.stats_area_height = self.pygame_target_window_height - self.frames_area_height
        self.stats_area_top_y = self.frames_area_height

        self.border_thickness = 15
        max_width_for_both_eyes = self.pygame_target_window_width - self.border_thickness

        if self.vr_render_height_original > 0 and self.vr_render_width_original > 0:
            aspect_ratio_vr = self.vr_render_width_original / self.vr_render_height_original
            self.monitor_eye_view_height = self.frames_area_height
            self.monitor_eye_view_width = int(self.monitor_eye_view_height * aspect_ratio_vr)
            if self.monitor_eye_view_width * 2 + self.border_thickness > self.pygame_target_window_width:
                self.monitor_eye_view_width = max_width_for_both_eyes // 2
                self.monitor_eye_view_height = int(self.monitor_eye_view_width / aspect_ratio_vr)
        else:
            self.monitor_eye_view_width = max_width_for_both_eyes // 2
            self.monitor_eye_view_height = int(self.frames_area_height * 0.8)

        if self.monitor_eye_view_height > self.frames_area_height:
            self.monitor_eye_view_height = self.frames_area_height
            if self.vr_render_width_original > 0 and self.vr_render_height_original > 0:
                aspect_ratio_vr = self.vr_render_width_original / self.vr_render_height_original
                self.monitor_eye_view_width = int(self.monitor_eye_view_height * aspect_ratio_vr)

        self.error_print_interval = 2.0
        self.last_error_print_time = 0
        self.show_detailed_stats_pygame = True

    def _initialize_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.pygame_target_window_width, self.pygame_target_window_height))
        pygame.display.set_caption("OptiX VR Monitor | S:Toggle Stats | ESC:Close")
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.Font(None, self.font_small_size)
        self.font_medium = pygame.font.Font(None, self.font_medium_size)
        self.font_large = pygame.font.Font(None, self.font_large_size)

        print(f"Pygame monitor initialized. Target Window: {self.pygame_target_window_width}x{self.pygame_target_window_height}")
        print(f"  Frames Area Height: {self.frames_area_height}, Stats Area Height: {self.stats_area_height}")
        print(f"  Monitor Eye View (scaled): {self.monitor_eye_view_width}x{self.monitor_eye_view_height} each")

    def _process_eye_frame_for_display(self, tensor_cpu, target_rect, eye_label, timings):
        # (This function's internal logic remains mostly the same as before)
        if tensor_cpu is not None and tensor_cpu.numel() > 0:
            try:
                frame_numpy = tensor_cpu.numpy()
                if frame_numpy.ndim != 3 or frame_numpy.shape[2] not in [3, 4]:
                    raise ValueError(f"Unexpected frame shape {frame_numpy.shape}")

                frame_numpy_float = frame_numpy.astype(np.float32)
                if frame_numpy_float.max() > 1.5:
                    frame_numpy_float = np.clip(frame_numpy_float / 255.0, 0.0, 1.0)
                else:
                    frame_numpy_float = np.clip(frame_numpy_float, 0.0, 1.0)

                frame_uint8 = (frame_numpy_float * 255).astype(np.uint8)
                channel_type = 'RGB' if frame_uint8.shape[2] == 3 else 'RGBA'

                surface_eye_orig = pygame.image.frombuffer(frame_uint8.tobytes(), (frame_uint8.shape[1], frame_uint8.shape[0]), channel_type)
                #surface_eye_orig = pygame.transform.flip(surface_eye_orig, False, True)
                surface_scaled = pygame.transform.scale(surface_eye_orig, (target_rect.width, target_rect.height))
                self.screen.blit(surface_scaled, target_rect.topleft)

                text_line_height = int(self.font_small.get_height() * 1.25)
                y_text_offset = target_rect.bottom - (text_line_height * 3) - 12

                def draw_text_on_frame(text, y_off, font=self.font_small, color=(230, 230, 230)):
                    text_surface = font.render(text, True, color)
                    shadow_surface = font.render(text, True, (20,20,20))
                    self.screen.blit(shadow_surface, (target_rect.left + 11, y_off + 1))
                    self.screen.blit(text_surface, (target_rect.left + 10, y_off))
                    return y_off + text_line_height

                y_text_offset = draw_text_on_frame(f"{eye_label} OptiX R:{timings.get('render_ms',0):.1f} Optix_copy:{timings.get('optix_copy_ms',0):.1f}", y_text_offset)
                y_text_offset = draw_text_on_frame(f"VR TexCopy: {timings.get('texture_copy_ms',0):.1f}ms", y_text_offset)
                if timings.get('gpu_to_cpu_ms', 0) > 0.01 or timings.get('cpu_to_gpu_ms', 0) > 0.01 :
                    y_text_offset = draw_text_on_frame(f"XCopy D2H:{timings.get('gpu_to_cpu_ms',0):.1f} H2D:{timings.get('cpu_to_gpu_ms',0):.1f}", y_text_offset)

            except Exception as e:
                current_time = time.time()
                if current_time - self.last_error_print_time > self.error_print_interval:
                    print(f"Pygame Error displaying {eye_label} eye: {e}")
                    self.last_error_print_time = current_time
                pygame.draw.rect(self.screen, (100,0,0), target_rect)
                error_text = self.font_small.render("Error", True, (255,255,255))
                self.screen.blit(error_text, (target_rect.centerx - error_text.get_width()//2, target_rect.centery - error_text.get_height()//2))
        else:
            pygame.draw.rect(self.screen, (50, 50, 50), target_rect)
            wait_text_content = f"Waiting for {eye_label}..."
            if tensor_cpu is not None and tensor_cpu.numel() == 0:
                wait_text_content = f"{eye_label} Empty"
            wait_text = self.font_medium.render(wait_text_content, True, (180,180,180))
            self.screen.blit(wait_text, (target_rect.centerx - wait_text.get_width()//2, target_rect.centery - wait_text.get_height()//2))


    def _render_loop(self):
        self._initialize_pygame()

        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                    if event.key == pygame.K_s:
                        self.show_detailed_stats_pygame = not self.show_detailed_stats_pygame
                        caption_stats = "ON" if self.show_detailed_stats_pygame else "OFF"
                        pygame.display.set_caption(f"OptiX VR Monitor | S:Stats ({caption_stats}) | ESC:Close")

            current_data_snapshot = {}
            with self.data_lock:
                current_data_snapshot = self.latest_monitor_data.copy()

            self.screen.fill((30, 30, 40))

            total_eye_view_width_with_border = self.monitor_eye_view_width * 2 + self.border_thickness
            eye_views_start_x = (self.pygame_target_window_width - total_eye_view_width_with_border) // 2
            eye_views_start_y = (self.frames_area_height - self.monitor_eye_view_height) // 2

            left_rect = pygame.Rect(eye_views_start_x, eye_views_start_y, self.monitor_eye_view_width, self.monitor_eye_view_height)
            right_rect = pygame.Rect(eye_views_start_x + self.monitor_eye_view_width + self.border_thickness, eye_views_start_y, self.monitor_eye_view_width, self.monitor_eye_view_height)

            left_timings = { "render_ms": current_data_snapshot.get("left_render_ms",0), "optix_copy_ms": current_data_snapshot.get("left_optix_copy_ms",0), "texture_copy_ms": current_data_snapshot.get("left_texture_copy_ms",0), "gpu_to_cpu_ms": current_data_snapshot.get("left_gpu_to_cpu_ms",0), "cpu_to_gpu_ms": current_data_snapshot.get("left_cpu_to_gpu_ms",0) }
            right_timings = { "render_ms": current_data_snapshot.get("right_render_ms",0), "optix_copy_ms": current_data_snapshot.get("right_optix_copy_ms",0), "texture_copy_ms": current_data_snapshot.get("right_texture_copy_ms",0), "gpu_to_cpu_ms": current_data_snapshot.get("right_gpu_to_cpu_ms",0), "cpu_to_gpu_ms": current_data_snapshot.get("right_cpu_to_gpu_ms",0) }

            self._process_eye_frame_for_display(current_data_snapshot.get("left_eye_cpu_tensor"), left_rect, "L", left_timings)
            self._process_eye_frame_for_display(current_data_snapshot.get("right_eye_cpu_tensor"), right_rect, "R", right_timings)

            if total_eye_view_width_with_border < self.pygame_target_window_width:
                border_rect_fill = pygame.Rect(eye_views_start_x + self.monitor_eye_view_width, eye_views_start_y, self.border_thickness, self.monitor_eye_view_height)
                pygame.draw.rect(self.screen, (20,20,25), border_rect_fill)

            # --- Stats Area (Bottom 25%) ---
            stats_bg_rect = pygame.Rect(0, self.stats_area_top_y, self.pygame_target_window_width, self.stats_area_height)
            pygame.draw.rect(self.screen, (45, 45, 55), stats_bg_rect)
            pygame.draw.line(self.screen, (90, 90, 100), (0, self.stats_area_top_y), (self.pygame_target_window_width, self.stats_area_top_y), 2)

            y_offset = self.stats_area_top_y + 10 # Start a bit lower for more top margin in stats area
            padding_x = 30

            line_h_medium = int(self.font_medium.get_height() * 1.25)
            line_h_large = int(self.font_large.get_height() * 1.25)
            line_h_small = int(self.font_small.get_height() * 1.25)

            vr_fps_text_content = f"VR FPS: {current_data_snapshot.get('vr_fps', 0.0):.1f}"
            vr_fps_surf = self.font_large.render(vr_fps_text_content, True, (230, 230, 180))
            self.screen.blit(vr_fps_surf, (padding_x, y_offset))

            main_loop_ms = current_data_snapshot.get('main_loop_time_ms', 0.0)
            loop_time_text_content = f"Main Loop: {main_loop_ms:.1f}ms"
            loop_time_surf = self.font_large.render(loop_time_text_content, True, (230, 230, 180))
            self.screen.blit(loop_time_surf, (self.pygame_target_window_width - loop_time_surf.get_width() - padding_x, y_offset))
            y_offset += line_h_large + 5 # Slightly less space after this top line

            if self.show_detailed_stats_pygame:
                # --- "System GPU Performance" title moved higher and centered ---
                stats_title_surf = self.font_large.render("System GPU Performance", True, (230, 230, 255))
                title_rect = stats_title_surf.get_rect(centerx=self.pygame_target_window_width // 2, top=y_offset)
                self.screen.blit(stats_title_surf, title_rect)
                y_offset += line_h_large + 10 # Space after title

                gpu_stats_list = current_data_snapshot.get("system_gpu_stats", [])
                if gpu_stats_list:
                    bar_width = int(150 * 1.3)
                    bar_height = int(20 * 1.3)

                    num_gpus_total = len(gpu_stats_list)
                    gpus_per_row = 1
                    if num_gpus_total > 1 and self.pygame_target_window_width > 1000: # If wide enough for 2 columns
                        gpus_per_row = 2

                    col_width_gpu_stats = self.pygame_target_window_width // gpus_per_row

                    initial_y_for_gpus_row = y_offset
                    max_gpu_block_height_this_row = 0

                    # --- Define consistent X positions for elements within each GPU block ---
                    # These are relative to the start of the GPU's column (block_x_start)
                    x_label_offset = 0
                    x_value_offset = self.font_small.size("Util: ")[0] + 10 # After "Util: "
                    # --- Increased space for RAM text and bar alignment ---
                    # Estimate max width for value text (e.g., "12345/24000MB")
                    max_mem_text_width = self.font_small.size("99999/99999MB")[0] + 10
                    x_bar_offset = x_value_offset + max_mem_text_width + 25 # Bar starts after the longest potential value string + more space
                    x_temp_offset = x_bar_offset + bar_width + 25 # Temp starts after bar + space

                    for i, gpu in enumerate(gpu_stats_list):
                        row_idx = i // gpus_per_row
                        col_idx = i % gpus_per_row

                        current_block_x_start = col_idx * col_width_gpu_stats + padding_x
                        current_block_y_start = initial_y_for_gpus_row + row_idx * (max_gpu_block_height_this_row + 25) # More space between rows

                        if col_idx == 0 : # Reset max height for a new row
                            max_gpu_block_height_this_row = 0

                        if current_block_y_start + (line_h_small * 2 + bar_height) > self.pygame_target_window_height - 10:
                            continue

                        gpu_name_text = self.font_medium.render(f"GPU {gpu['id']}: {gpu['name']}", True, (220,220,220)) # Full name
                        self.screen.blit(gpu_name_text, (current_block_x_start, current_block_y_start))
                        current_y_stat_line = current_block_y_start + line_h_medium + 2

                        # GPU Util
                        util_val = gpu.get('gpu_util', 0.0)
                        util_color = (0,255,0) if util_val < 50 else (255,255,0) if util_val < 80 else (255,0,0)
                        self.screen.blit(self.font_small.render("Util:", True, (200,200,200)), (current_block_x_start + x_label_offset, current_y_stat_line))
                        self.screen.blit(self.font_small.render(f"{util_val:.0f}%", True, util_color), (current_block_x_start + x_value_offset, current_y_stat_line))
                        pygame.draw.rect(self.screen, (100,100,100), (current_block_x_start + x_bar_offset, current_y_stat_line + (self.font_small.get_height() - bar_height)//2, bar_width, bar_height))
                        pygame.draw.rect(self.screen, util_color, (current_block_x_start + x_bar_offset, current_y_stat_line + (self.font_small.get_height() - bar_height)//2, int(bar_width * util_val/100), bar_height))

                        temp_val = gpu.get('temperature', 0.0)
                        temp_color = (0,255,0) if temp_val < 70 else (255,255,0) if temp_val < 80 else (255,0,0)
                        self.screen.blit(self.font_small.render(f"T:{temp_val:.0f}C", True, temp_color), (current_block_x_start + x_temp_offset, current_y_stat_line))
                        current_y_stat_line += line_h_small + 2

                        # Memory Util
                        mem_used = gpu.get('memory_used',0.0)
                        mem_total = gpu.get('memory_total',1.0)
                        mem_util_val = (mem_used / mem_total * 100) if mem_total > 0 else 0
                        mem_color = (0,255,0) if mem_util_val < 50 else (255,255,0) if mem_util_val < 80 else (255,0,0)
                        self.screen.blit(self.font_small.render("Mem:", True, (200,200,200)), (current_block_x_start + x_label_offset, current_y_stat_line))
                        mem_value_text = f"{mem_used:.0f}/{mem_total:.0f}MB"
                        self.screen.blit(self.font_small.render(mem_value_text, True, mem_color), (current_block_x_start + x_value_offset, current_y_stat_line))
                        # Bar X position is already determined by x_bar_offset for alignment
                        pygame.draw.rect(self.screen, (100,100,100), (current_block_x_start + x_bar_offset, current_y_stat_line + (self.font_small.get_height() - bar_height)//2, bar_width, bar_height))
                        pygame.draw.rect(self.screen, mem_color, (current_block_x_start + x_bar_offset, current_y_stat_line + (self.font_small.get_height() - bar_height)//2, int(bar_width * mem_util_val/100), bar_height))

                        # Calculate the height of the current GPU block for row wrapping
                        current_block_height = (current_y_stat_line + line_h_small) - (initial_y_for_gpus_row + row_idx * (max_gpu_block_height_this_row + 20))
                        max_gpu_block_height_this_row = max(max_gpu_block_height_this_row, current_block_height)


                else: # if not gpu_stats_list
                    no_stats_text = self.font_medium.render("GPU stats via nvidia-smi unavailable", True, (150,100,100))
                    no_stats_rect = no_stats_text.get_rect(centerx=self.pygame_target_window_width // 2, top=y_offset)
                    self.screen.blit(no_stats_text, no_stats_rect)

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        print("Pygame monitor thread stopped.")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._render_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("Warning: PygameMonitor thread did not join cleanly.")

    def update_data(self, new_data_dict):
        with self.data_lock:
            self.latest_monitor_data.update(new_data_dict)

    def should_quit(self):
        return not self.is_running

# --- Standalone Test Block ---
if __name__ == "__main__":
    print("Running PygameMonitor standalone test: 75% Frames / 25% Stats layout, Improved Alignment.")

    mock_vr_width_original = 800
    mock_vr_height_original = 600

    test_monitor_width = 1920
    test_monitor_height = 1080

    monitor = PygameMonitor(mock_vr_width_original, mock_vr_height_original,
                            test_monitor_width, test_monitor_height)
    monitor.start()

    def create_mock_frame(width, height, r_offset=0.0, g_offset=0.0, b_offset=0.0):
        r = torch.linspace(0 + r_offset, 0.5 + r_offset, width)
        g = torch.linspace(0 + g_offset, 0.5 + g_offset, height)
        grid_r, grid_g = torch.meshgrid(r, g, indexing='xy')
        b_channel = torch.ones_like(grid_r) * (0.2 + b_offset)
        frame_tensor = torch.stack([grid_r.T, grid_g.T, b_channel.T], dim=2)
        return torch.clip(frame_tensor, 0, 1)

    try:
        count = 0
        last_gpu_stats_update_time = 0
        gpu_stats_update_interval = 1.0
        mock_sys_gpu_stats = []

        while monitor.is_running:
            time.sleep(0.033)
            count +=1

            current_time_test = time.time()
            if current_time_test - last_gpu_stats_update_time > gpu_stats_update_interval:
                mock_sys_gpu_stats = get_gpu_info()
                if not mock_sys_gpu_stats:
                    mock_sys_gpu_stats = [
                        {'id': 0, 'name': 'NVIDIA GeForce RTX 3090', 'gpu_util': random.randint(30,70),
                         'memory_used': random.randint(8000,15000), 'memory_total': 24576, # Example total
                         'temperature': random.randint(50,75)},
                        {'id': 1, 'name': 'NVIDIA GeForce RTX 4090 Laptop GPU', 'gpu_util': random.randint(40,85),
                         'memory_used': random.randint(6000,12000), 'memory_total': 16384, # Example total
                         'temperature': random.randint(60,80)}
                    ] if random.random() > 0.1 else []
                last_gpu_stats_update_time = current_time_test

            mock_data = {
                "left_eye_cpu_tensor": create_mock_frame(mock_vr_width_original, mock_vr_height_original, r_offset=random.random()*0.5, b_offset=0.1* (count % 5)),
                "right_eye_cpu_tensor": create_mock_frame(mock_vr_width_original, mock_vr_height_original, g_offset=random.random()*0.5, b_offset=0.1* (count % 5)),
                "left_render_ms": 5.0 + random.random() * 2, "left_optix_copy_ms": 0.5 + random.random() * 0.2,
                "left_texture_copy_ms": 1.0 + random.random() * 0.5, "left_gpu_to_cpu_ms": 0.1 + random.random() * 0.1,
                "left_cpu_to_gpu_ms": 0.05 + random.random() * 0.05,
                "right_render_ms": 5.2 + random.random() * 2, "right_optix_copy_ms": 0.6 + random.random() * 0.2,
                "right_texture_copy_ms": 1.1 + random.random() * 0.5, "right_gpu_to_cpu_ms": 0.12 + random.random() * 0.1,
                "right_cpu_to_gpu_ms": 0.06 + random.random() * 0.05,
                "total_to_cpu_ms": 0.22 + random.random() * 0.2, "total_from_cpu_ms": 0.11 + random.random() * 0.1,
                "vr_fps": 85.0 + random.random() * 10, "main_loop_time_ms": 11.1 + random.random() * 1,
                "system_gpu_stats": mock_sys_gpu_stats
            }
            monitor.update_data(mock_data)

            if not monitor.is_running:
                break

    except KeyboardInterrupt:
        print("Standalone test interrupted.")
    finally:
        print("Requesting standalone monitor to stop...")
        monitor.stop()
        print("Standalone test finished.")