// Fragment shader
#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

// Output color along with alpha value
void main() {
    outColor = vec4(fragColor, 1.0);
}


/*  We will not deal with materials or lighting in our scenes until A2. However, showing only vertex colors is quite "flat" looking, so display every scene as if lit by a hemispherical light in direction (0,0,1) with energy (1,1,1).
 
 In other words, your fragment shader should do something along these lines:

     vec3 light = mix(vec3(0,0,0), vec3(1,1,1),
         dot(normal, vec3(0,0,1)) * 0.5 + 0.5);
     outColor = vec4(light * color, 1.0);
  Basic hemispherical lighting equation in glsl syntax, where: normal is the per-pixel normal (remember to normalize after interpolation!); color is the interpolated vertex color; and outColor is the value that gets written to the framebuffer.
 */
