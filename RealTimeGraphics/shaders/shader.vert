// Vertex shader processes each incoming vertex
// Input: attributes, like model space position, color, normal and texture coordinates
// Output: final position in clip coordinates(4D vector, normalised by last elem) & attributes to be passed to fragment shader(color, texture coord)
// Normalized device coordinates are homogeneous coordinates that map the framebuffer to a [-1, 1] by [-1, 1] coordinate system
#version 450

// uniform buffer object descriptor: model, view and projection matrices
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// vertex attributes: properties that are specified per-vertex in the vertex buffer
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

// Invoked for every vertex
// gl_VertexIndex: index of current vertex
void main() {
    // use the transformations to compute the final position in clip coordinates
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    
    // Pass per-vertex colors to frag shader -> output interpolated values to framebuffer
    fragColor = inColor;
}
