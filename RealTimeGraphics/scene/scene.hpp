#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/* 
Defines the classes for scene graph structure that derives data from a scene'72 file
Contains 5 object types: SCENE, NODE, MESH, CAMERA, DRIVER
*/

/*
 * Scene represents a 3D scene with a scene graph.
 *
 * Resources in the scene are held with std::shared_ptr<>'s.
 * Resources refer to other resources via std::weak_ptr<>'s.
 *
 * NOTE: except for parent pointers in Transforms,
 *  which can be nullptr, any weak_ptr<> held by a resource
 *  must point to something else held via shared_ptr<> in the
 *  same scene, such that weak_ptr< >::lock() will always
 *  succeed.
 *
 *  So you generally don't need to write code like:
 *   if (auto val = ptr->lock()) { val->do_thing(); }
 *  And can instead just write:
 *   ptr->lock()->do_thing();
 *
 */

// Scene: represents a scene graph in a 3D scene, which defines global properties of the scene
class Scene {
  std::string name; // name of scene
  std::vector<Node> roots; // array of references to nodes at which to start drawing the scene
  std::vector<Camera> cameras; // direct access to cameras, map from idx to camera
  std::vector<Driver> drivers; // direct access to drivers, map from idx to driver
  std::vector<Node> drivers; // direct access to nodes, map from idx to driver
    // list of cameras, meshes, nodes, drivers
};

// Node: structure of a scene is determined by a graph of transformation nodes
/*"
	"name":"bottom",
	"translation":[0,0,0],
	"rotation":[0,0,0,1],
	"scale":[1,1,1],
	"children":[2,3,4,5],
	"camera":7,
	"mesh":2*/
class Node {
  std::string name; // name of node
  glm::vec3 translation; // translation":[tx,ty,tz] (optional; default is [0,0,0]) -- the translation part of the node's transform, as a 3-element array of numbers
  glm::vec4 rotation; //rotation":[rx,ry,rz,rw] (optional; default is [0,0,0,1]) -- the rotation part of the node's transform, as a unit quaternion (where rw is the scalar part of the quaternion)
  glm::vec3 scale; // [sx,sy,sz] (optional; default is [1,1,1]) -- the scale part of the node's transform, as a 3-element array of axis-aligned scale factors.
  std::vector<Node> children; // "children":[...] (optional; default is []) -- array of references to nodes which should be instanced as children of this transformation, map from idx to child node
  Mesh mesh; // "mesh":i (optional) -- reference to a mesh to instance at this node.
  Camera camera; // "camera":i (optional) -- reference to a camera to instance at this node.

/* 
The transformation from the local space of a node to the local space of its parent node is given by applying its scale, rotation, and translation values (in that order): M(parent from local) = T*R*S

Note: the structure of the graph on node objects induced by their children arrays is not restricted by this specification. Thus, e.g., there may be multiple paths from the root of the scene to a given node. (Effectively, instancing entire transformation sub-trees.) However, implementations may choose to reject files containing cyclic transformation graphs.
*/

// parent from local
// local from parent
// traverse
};

// MeshAttributes: data streams used to define the mesh vertices
// Attribute stream names should follow the naming convention used by glTF (e.g., using "POSITION" for the position stream, "NORMAL" for vertex normals, "COLOR" for vertex colors, and so on). However, stream formats are not restricted by the glTF conventions
class MeshAttributes {
/* 

"src":"..." (required) -- file to read data from. Note that the path is specified relative to the ".s72" file.
"offset":N (required) -- byte offset from the start of the file for the first element of this attribute stream.
"stride":S (required) -- bytes between the starts of subsequent elements of this attribute stream.
"format":"..." (required) -- format of the stored attribute. Valid strings are VkFormat identifiers without the prefix, e.g., R32G32B32_SFLOAT.

 use the R32G32B32_FLOAT and R8G8B8A8_UNORM formats, https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#features-required-format-support


If a mesh contains an indices property, it is an indexed mesh -- its vertex stream must be constructed by reading indices from the specified data stream and using these to access the vertex stream. Otherwise its vertex stream must be drawn sequentially from the attributes array. (See Programmable Primitive Shading for the Vulkan specification's lists of indexed and non-indexed drawing commands.)

Index streams are defined similarly to attribute streams with two differences:

"format" must be a VkIndexType identifier without the common prefix (e.g., "UINT32").
"stride" must be omitted -- indices are always tightly packed.
In index streams, the all-ones index (e.g., 0xffffffff for "format":"UINT32" streams) is used to indicate primitive restart.

Note about index formats: An scene'72 loader must support "UINT32" format, and may support other formats.

*/
};

// Mesh: frawable geometry in the scene
/* "type":"MESH",
	"name":"cube",
	"topology":"TRIANGLE_LIST",
	"count":12,
	"indices": { "src":"cube.b72", "offset":576, "format":"UINT32" },
	"attributes":{
		"POSITION":{ "src":"cube.b72", "offset":0,  "stride":28, "format":"R32G32B32_SFLOAT" },
		"NORMAL":  { "src":"cube.b72", "offset":12, "stride":28, "format":"R32G32B32_SFLOAT" },
		"COLOR":   { "src":"cube.b72", "offset":24, "stride":28, "format":"R8G8B8A8_UNORM" }
	}
  */
class Mesh {
/* 
"topology":"..." (required) -- the primitive type and layout used in the mesh. Valid values are VkPrimitiveTopology identifiers without the prefix (e.g., "TRIANGLE_LIST").
"count":N (required) -- the number of vertices in the mesh.
"indices":{ ... } (optional -- if specified, a data stream containing indices for indexed drawing commands.
"attributes":{ ... } (required) -- named data streams containing the mesh attributes.*/

};

// Camera: define projection parameters of cameras in the scene
class Camera {
/*
	"name":"main view",
	"perspective":{
		"aspect": 1.777,
		"vfov": 1.04719,
		"near": 0.1,
		"far":10.0
	}
},

"perspective":{...} (optional) -- defines that the camera uses perspective projection. Contains child properties:
"aspect":a (required) -- image aspect ratio (width / height).
"vfov":r (required) -- vertical field of view in radians.
"near":z (required) -- near clipping plane distance.
"far":z (optional) -- far clipping plane distance; if omitted, use an infinite perspective matrix.
Scene'72 cameras look down their local -z axis with +y being upward and +x being rightward in the image plane.

Camera objects must define some projection; so even though "perspective" is marked as "optional" above it is de-facto required unless you decide to add (say) "orthographic" cameras.

If rendering through a camera that does not match the output image aspect ratio, a scene'72 viewer should letter- or pillar-box the output image.
*/

};

// Driver: objects drive (animate) properties of other objects
class Driver {
/* 
	"name":"camera move",
	"node":12,
	"channel":"translation",
	"times":[0, 1, 2, 3, 4],
	"values":[0,0,0, 0,0,1, 0,1,1, 1,1,1, 0,0,0],
	"interpolation":"LINEAR"

"node":i (required) -- reference to the node whose property should be animated by this driver.
"channel":"..." (required) -- name of an animation channel; implies a data width (see below).
"times":[...] (required) -- array of numbers giving keyframe times.
"values":[...] (required) -- array of numbers giving keyframe values.
"interpolation":"..." (optional; default is "LINEAR") -- interpolation mode for the data (see below).
The values in the values array are grouped into 1D-4D vectors depending on the channel type and interpolation scheme. For example, a 3D channel with 
 times will have 
 values, which should be considered as 
 3-vectors.

The possible channel values and their meanings are as follows:

3D channels: "translation", "scale". Meaning: set the associated component of the target node.
4D channels: "rotation". Meaning: set the rotation of the target node as a quaternion.
The meaning of interpolation is as follows:

"STEP" the output value in the middle of a time interval is the value at the beginning of that interval.
"LINEAR" the output value in the middle of a time interval is a linear mix of the starting and ending values.
"SLERP" the output value in the middle of a time interval is a "spherical linear interpolation" between the starting and ending values. (Doesn't make sense for 1D signals or non-normalized signals.)
Extrapolation is always constant.

The effect of applying driver objects should be as if the objects are applied in the order they appear in the file. I.e., later driver objects may override earlier driver objects that drive the same properties.
  */
};
