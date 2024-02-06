//A "Somewhat Eager JSON Parser" that parses and converts files
//upon loading into some lists of numbers, objects, bools, nulls;
//then provides a generic "value" handle to the root.

// A JSON parser that parses scene .72 format (https://github.com/15-472/s72) for animated 3D scenes
// UTF8-encoded JSON
// scene'72 scenes are z-up, lengths are meters, and times are seconds
// Object types: scene, node, mesh, camera, driver
// Takes inspiration from SEJP (https://github.com/ixchow/sejp)

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <memory>

