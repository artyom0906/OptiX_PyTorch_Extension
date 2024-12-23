import pygame
import moderngl
import numpy as np
from pygltflib import GLTF2
from pyrr import Matrix44
import base64
import os

class GLTFModelRenderer:
    def __init__(self, gltf_path, ctx):
        self.ctx = ctx
        self.gltf = GLTF2().load(gltf_path)
        # For simplicity, this example will assume the GLTF file has positions only
        self.vertices = []
        self.indices = []

        # Extract mesh data from the GLTF
        for mesh in self.gltf.meshes:
            for primitive in mesh.primitives:
                accessor = self.gltf.accessors[primitive.attributes.POSITION]
                buffer_view = self.gltf.bufferViews[accessor.bufferView]
                buffer = self.gltf.buffers[buffer_view.buffer]
                if isinstance(buffer.uri, str) and buffer.uri.startswith("data:"):
                    # Handle embedded base64 data
                    data_str = buffer.uri.split(",", 1)[1]
                    buffer_data = base64.b64decode(data_str)
                else:
                    # Load the buffer data from an external file
                    buffer_path = os.path.join(os.path.dirname(gltf_path), buffer.uri)
                    with open(buffer_path, "rb") as f:
                        buffer_data = f.read()
                data = np.frombuffer(buffer_data, np.float32)
                self.vertices.extend(data.tolist())

        # Create OpenGL buffer
        self.vbo = self.ctx.buffer(np.array(self.vertices, dtype='f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                void main() {
                    gl_Position = projection * view * model * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
                }
            '''
        ), self.vbo, 'in_position')

    def render(self, model_matrix, view_matrix, projection_matrix):
        self.vao.program['model'].write(model_matrix.astype('f4').tobytes())
        self.vao.program['view'].write(view_matrix.astype('f4').tobytes())
        self.vao.program['projection'].write(projection_matrix.astype('f4').tobytes())
        self.vao.render()

def main():
    pygame.init()
    window_size = (800, 600)
    pygame.display.set_mode(window_size, pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    # Create OpenGL context using ModernGL
    ctx = moderngl.create_context()

    # Load and prepare GLTF model
    model_renderer = GLTFModelRenderer("models/elevator/scene.gltf", ctx)

    # Create view and projection matrices
    model_matrix = Matrix44.identity()
    view_matrix = Matrix44.look_at(
        eye=[3.0, 3.0, 3.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0]
    )
    projection_matrix = Matrix44.perspective_projection(45.0, window_size[0] / window_size[1], 0.1, 100.0)

    # Movement controls
    rotation_angle = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            rotation_angle += 1.0
        if keys[pygame.K_RIGHT]:
            rotation_angle -= 1.0
        if keys[pygame.K_UP]:
            view_matrix = Matrix44.look_at(
                eye=[3.0, 3.0, 3.0 - 0.1],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0]
            )
        if keys[pygame.K_DOWN]:
            view_matrix = Matrix44.look_at(
                eye=[3.0, 3.0, 3.0 + 0.1],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0]
            )

        # Update model matrix with rotation
        model_matrix = Matrix44.from_y_rotation(np.radians(rotation_angle))

        # OpenGL rendering
        ctx.clear(0.2, 0.4, 0.6)
        model_renderer.render(model_matrix, view_matrix, projection_matrix)

        # Swap buffers
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
