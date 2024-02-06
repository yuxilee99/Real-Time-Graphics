/*
Math library that contains operations for Mat4, Vec4, Vec3, Vec2, and Quaternions
References math library structure of glm and Scotty3D's math library
*/

// Math library header files
#include "mat4.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "quat.hpp"

#include <algorithm>
#include <cmath>

// Common constants
#define EPS_F 0.00001f
#define PI_F 3.14159265358979323846264338327950288f
#define PI_D 3.14159265358979323846264338327950288
#define Radians(v) ((v) * (PI_F / 180.0f))
#define Degrees(v) ((v) * (180.0f / PI_F))