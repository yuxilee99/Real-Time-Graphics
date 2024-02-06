#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <fstream>
#include <array>
#include <chrono>

// window dimensions
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2; // prevent CPU from getting too far ahead of GPU

// Validation layers: optional components that hook into Vulkan fn valls to apply additional ops
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// List of required device extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

// Proxy fn that passes debug messenger callback struct to vkCreateDebugUtilsMessengerEXT
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else return VK_ERROR_EXTENSION_NOT_PRESENT;
}

// Proxy fn that cleans up CreateDebugUtilsMessengerEXT
void DestroyDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) func(instance, debugMessenger, pAllocator);
}

// struct for queues
// uses optional to determine if queue exists
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; // optional has no value until you assign something to it
    std::optional<uint32_t> presentFamily;// ensure device can present images to surface we created
    
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Swap chain: check if compatible with window surface with 3 properties:
// - basic surface capabilities
// - surface formats (pixel format, color space)
// - avaliable presentation modes
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Vertex shader struct
struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    
    // Vertex binding: describes at which rate to load data from memory throughout the vertices
    // Specifies # of bytes between data entries, whether to move to the next data entry after each vertex or after each instance
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // per vertex data
        
        return bindingDescription;
    }
    
    // Attribute description: describes how to extract a vertex attribute from a chunk of vertex data originating from a binding description
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        // Position attribute
        attributeDescriptions[0].binding = 0; // from which binding the per-vertex data comes
        attributeDescriptions[0].location = 0; // location directive of the input in the vertex shader, input in the vertex shader w/ location 0
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT; // describes the type of data for the attribute, byte size of attribute data
        attributeDescriptions[0].offset = offsetof(Vertex, pos); //  number of bytes since the start of the per-vertex data to read from
        
        // Color attribute
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

// uniform buffer object descriptor
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

// Triangle vertices const
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

// Triangle indices const
const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};


class RenderApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    GLFWwindow* window;
    
    VkInstance instance;
    
    VkDebugUtilsMessengerEXT debugMessenger;
    
    VkSurfaceKHR surface;
    
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    
    VkQueue graphicsQueue; // handle to graphics queue
    
    VkQueue presentQueue;
    
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    
    std::vector<VkImageView> swapChainImageViews;
    
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    
    std::vector<VkFramebuffer> swapChainFramebuffers; // hold framebuffers
    
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    
    // Semaphores are used to specify the execution order of operations on the GPU
    // Fences are used to keep the CPU and GPU in sync with each-other
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    bool framebufferResized = false;
    
    uint32_t currentFrame = 0;
    
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    
    // initialises window from GLFW
    void initWindow() {
        glfwInit();
        
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // initialise GLFW library without creating OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // disable window resizing
        
        // create a window: width, height, window title, monitor specifation, OpenGl related param
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }
    
    // detect resizes
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<RenderApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
    
    // calls initiate functions for Vuklan objs
    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }
    
    // starts rendering frames
    void mainLoop() {
        // iterates until window is closed
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents(); // checks for events
            drawFrame();
        }
        
        // Wait for the logical device to finish operations before exiting
        vkDeviceWaitIdle(device);
    }
    
    void cleanupSwapChain() {
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }
    
    // deallocates/frees resources used
    // Every Vulkan obj created must be explicitly destroyed when no longer needed
    void cleanup() {
        cleanupSwapChain();
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        
        vkDestroyDescriptorPool(device, descriptorPool, nullptr); // descriptor sets automatically freed when descriptor pool is destroyed
    
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
        
        vkDestroyCommandPool(device, commandPool, nullptr);
        
        vkDestroyDevice(device, nullptr);

        // Debug messenger
        if (enableValidationLayers) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        // GFLW resources
        glfwDestroyWindow(window);
        glfwTerminate();
        
    }
    
    // Recreating swap chain: calls createSwapChain and all of the creation functions for the objects that depend on the swap chain/window size.
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height); // size is already correct
        // pause until the window is in the foreground again
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);
        
        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }
    
    // Initialise Vulkan library by creating an instance
    // Connection b/w application & Vuklan library
    // General obj creation: ptr to struct w/ create info, ptr to custom allocater callbacks, ptr to varable that stores handle to new obj
    void createInstance() {
        // Check if validation layers are avaliable
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not avaliable!");
        }
        
        // Provides application information
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Renderer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        
        // Tells Vulkan driver which global extensions & validation layers to use
        // Global - apply to entire program, NOT specific device
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        // Specify desired global extensions bc Vulkan platform agnostic API
        auto extensions = getRequiredExtensions();
//        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
//        createInfo.ppEnabledExtensionNames = extensions.data();
        
        // Incompatible driver case for MacOS
        std::vector<const char*> requiredExtensions;
        
        for (uint32_t i = 0; i < extensions.size(); i++) {
            requiredExtensions.emplace_back(extensions.data()[i]);
        }
        
        requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        requiredExtensions.emplace_back("VK_KHR_get_physical_device_properties2");
        
        createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();
        
        createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        
        
        // Determine global validation layers to enable
        // Include validation layers if enabled
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            
            // Create debug callback message information
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }
        
        // Create an instance w/ Vulkan
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }
    
    // checks if all of the requested layers are avaliable
    bool checkValidationLayerSupport() {
        // list all avaliable layers
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        
        std::vector<VkLayerProperties> avaliableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, avaliableLayers.data());
        
        // check if all layers in validationLayers exist in avaliableLayers
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            
            for (const auto& layerProperties : avaliableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            
            if (!layerFound) return false;
        }
        return true;
    }
    
    // Handles explicit callback debug messages from validation layers
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        
        // Debug messenger extension conditionally added
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }
    
    // Debug callback fn: returns true, if Vulkan call that triggered validation layer message should be aborted
    // pCallbackData: refers to struct containing detail of message
    // pUserData: ptr that was specified during setup of callback, allows you to pass data into it
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
    
    // Populates messenger create info for debug callback messenger
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr;
    }
    
    // Tells Vulkan about callback message
    void setupDebugMessenger() {
        if (!enableValidationLayers) return;
        
        // Populate create info
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);
        
        // Create extension obj if avaliable
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }
    
    // Pick graphics card/physical device
    // Note: current implementation picks first suitable device, can also give devices a score & pick highest one
    void pickPhysicalDevice() {
        // List graphics cards
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        // Case: no physical devices
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        // Check if device suitable
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }
        
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        
    }
    
    // Check if physical device is suitable for our ops
    bool isDeviceSuitable(VkPhysicalDevice device) {
//        // basic properties: name, type, supported Vulkan version
//        VkPhysicalDeviceProperties deviceProperties;
//        vkGetPhysicalDeviceProperties(device, &deviceProperties);
//
//        // features: texture compression, 64 bit floats, multi viewport rendering
//        VkPhysicalDeviceFeatures deviceFeatures;
//        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        
//        // Assume only usable for graphics cards that support geometry shaders
//        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;\
        
        // Ensure device can process cmds we want
        QueueFamilyIndices indices = findQueueFamilies(device);
        
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        
        // Verify that swap chain support is adequate:
        // at least one supported image format and one supported presentation mode given the window surface we have
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        
        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }
    
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        
        // Assign index to queue families that could be found
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        // Find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            // queue family needs to be able to present window surface
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) indices.presentFamily = i;
            
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;
            if (indices.isComplete()) break;
            i++;
        }
        
        return indices;
    }
    
    void createLogicalDevice() {
        // Describe # queues we want for single queue family
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()}; // create queue from both families
        
        // Can assign priorities to queues to influence scheduling of cmd buffer [0,1]
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        
        // Specify set of device features
        VkPhysicalDeviceFeatures deviceFeatures{};
        
        // Create logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        
        // queue creation info + device features structs
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        
        createInfo.pEnabledFeatures = &deviceFeatures;
        
        // Specify extensions + validation layers, device specific
        // Incompatible driver case for MacOS
        std::vector<const char*> requiredDeviceExtensions;
        
        for (uint32_t i = 0; i < deviceExtensions.size(); i++) {
            requiredDeviceExtensions.emplace_back(deviceExtensions.data()[i]);
        }
        
        
        requiredDeviceExtensions.emplace_back("VK_KHR_portability_subset");
//        requiredDeviceExtensions.emplace_back("VK_KHR_get_physical_device_properties2");
        
        createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();
        
        
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else createInfo.enabledLayerCount = 0;
        
        // Instantiate logical device
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
        
        // Retrieve queue handles for each queue family
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }
    
    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }
    
    // Enumerate extensions and check if required extensions are among them
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        
        std::vector<VkExtensionProperties> avaliableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, avaliableExtensions.data());
        
        // Use set of strings to represent unconfirmed required extensions
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        
        for (const auto& extension : avaliableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }
    
    // Populate SwapChainSupportDetails struct
    // To find the right settings for the best possible swap chain need to determine:
    // - Surface format (color depth).
    // - Presentation mode (conditions for "swapping" images to the screen)
    // - Swap extent (resolution of images in swap chain)
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        
        // Query supported surface formats, determine support capabilities
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        
        if (formatCount != 0) {
            details.formats.resize(formatCount); // vector is resized to hold all the available formats
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        
        // Query supported presentation modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount); // vector is resized to hold all the available formats
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        
        return details;
    }
    
    // Each VkSurfaceFormatKHR entry contains a format and a colorSpace member
    // format: specifies the color channels and types
    // colorSpace: indicates if the SRGB color space is supported with VK_COLOR_SPACE_SRGB_NONLINEAR_KHR flag
    // Picks first avaliable format that satisfies
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& avaliableFormats) {
        for (const auto& avaliableFormat : avaliableFormats) {
            if (avaliableFormat.format == VK_FORMAT_B8G8R8_SRGB && avaliableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return avaliableFormat;
            }
        }
        return avaliableFormats[0];
    }
    
    // Presentation mode represents the actual conditions for showing images to the screen
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& avaliablePresentModes) {
        // Instead of blocking the application when the queue is full,
        // the images that are already queued are simply replaced with the newer ones
        // Triple buffering: used to render frames as fast as possible, avoiding tearing, fewer latency issues than standard vertical sync
        for (const auto& avaliablePresentMode : avaliablePresentModes) {
            if (avaliablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return avaliablePresentMode;
            }
        }
        // display takes an image from the front of the queue when the display is refreshed
        // program inserts rendered images at the back of the queue
        // If the queue is full then the program has to wait
        // similar to vertical sync, display is refreshed is known as "vertical blank".
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    
    // Swap extent: resolution of the swap chain images
    // almost always exactly equal to the resolution of the window that we’re drawing to in pixels
    // Swap chain extent must be specified in pixels
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // Match the resolution of the window by setting the width and height in the currentExtent
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        
        // Pick the resolution that best matches the window within the minImageExtent and maxImageExtent bounds
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        
        VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        
        // bound width/height b/w the allowed min/max extents supported by the implementation
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        
        return actualExtent;
    }
    
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1; // # images in the swapchain
        // Ensure dont exceed the max # of images, where 0 indicates there is no maximum
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        
        // Fill in swap chain struct
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1; // specifies the amount of layers each image consists of, always 1 unless you are developing a stereoscopic 3D application
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // bit field specifies kind of operations use the images in the swap chain for, color attachment
        
        // Specify how to handle swap chain images that will be used across multiple queue families
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        
        // Graphics queue family is different from the presentation queue
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; // Images can be used across multiple queue families without explicit ownership transfers
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // image is owned by one queue family at a time, ownership must be explicitly transferred, best performance.
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }
        
        // If supported, specify if certain transform should be applied to images in swap chain
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        
        // If alpha channel should be used for blending with other windows in window system
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // ignores alpha channel
        
        createInfo.presentMode = presentMode;
        
        // If clipped member == true, dont care about color of pixels obscured bc another window in front
        createInfo.clipped = VK_TRUE;

        // Reference to old swapchain: needed if swapchain is invalid or unoptimised while app is running & swap chain needs to be recreated
        createInfo.oldSwapchain = VK_NULL_HANDLE; // assume we have one swapchain
        
        // Create the swapchain
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swapchain!");
        }
        
        // Retrieve swapchain handles: query final # of images, resize container, call to retrieve handles
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        
        // Store format, extent for swapchaim images
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }
    
    // Creates basic image view for every image in swapchain, to be used as color targets later
    // Create a VkImageView object: view into an image, describes how to access the image and which part of the image to access
    void createImageViews() {
        // Resize list to fit all image views
        swapChainImageViews.resize(swapChainImages.size());
        
        // Create image views
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            
            // Specify how image data should be interpreted
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // can treat img as 1D, 2D, 3D textures, cube maps
            createInfo.format = swapChainImageFormat;
            
            // Component field: lets you swizzle color channels(adjust the mapping)
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            
            // subresourceRange: describes image purpose & what part of image should be accessed
            // Here used as color targets w/o mipmapping levels/multiple layers
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            
            // Create the image view
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views");
            }
        }
    }
    
    void createGraphicsPipeline() {
        // Load bytecode of 2 shaders
        auto vertShaderCode = readFile("/Users/yuxi01px2018/Documents/RealTimeGraphics/RealTimeGraphics/RealTimeGraphics/shaders/vert.spv");
        auto fragShaderCode = readFile("/Users/yuxi01px2018/Documents/RealTimeGraphics/RealTimeGraphics/RealTimeGraphics/shaders/frag.spv");
        // Shaders are loaded correctly if size of the buffers match the actual file size in bytes
//        std::cout << "buffer sizes: " << vertShaderCode.size() << ", " << fragShaderCode.size() << std::endl;
        
        // Shader module: thin wrapper around shader bytecode loaded from file & fn defined in it
        // Stored as local b/c compilation and linking of the SPIR-V bytecode to machine code for execution by the GPU happens after graphics pipeline is created
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        
        // Shader stage creation: assign shader to speciic pipeline stage
        // Vertex shader struct
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // pipeline stage of shader
        
        // possible to combine multiple frag shaders into a single shader module & use different entry points to differentiate their behaviors
        vertShaderStageInfo.module = vertShaderModule; // module containing code
        vertShaderStageInfo.pName = "main"; // fn to invoke, entrypoint
        
        // Fragment shader struct
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule; // module containing code
        fragShaderStageInfo.pName = "main"; // fn to invoke, entrypoint
        
        // Create array with vertex, frag shader structs
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        
        // Vertex info: describe format of vert data to be passed to vert shader
        // Bindings: spacing between data and whether the data is per-vertex or per-instance (see instancing)
        // Attribute descriptions: type of the attributes passed to the vertex shader, which binding to load them from and at which offset
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        
        // Input assembly: describes kind of geometry will be drawn from verts & if prim restart should be enabled
        // Drawing triangles
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        
        // Viewport: describes the region of the framebuffer that the output will be rendered to, typically (0, 0) to (width, height), define the transformation from the image to the framebuffer
        // Scissor rectangles: in which regions pixels will actually be stored, pixels outside will be discarded by the rasterizer
        // viewport & scissor rectangle can either be specified as a static in pipeline or dynamic state set in the command buffer
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        
        // Rasterizer: takes geometry shaped by vertices from the vertex shader and turns it into fragments to be colored by the frag shader
        // Performs depth testing, face culling, scissor test
        // Can be configured to output fragments that fill entire polygons or just the edges (wireframe rendering)
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE; // if true, fragments beyond near and far planes are clamped instead of discarded
        rasterizer.rasterizerDiscardEnable = VK_FALSE; // if true, geometry never passes through rasterizer stage, disables output to framebuffer
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // determine how frags generated for geometry
        rasterizer.lineWidth = 1.0f; // thickness of lines in terms of number of fragments
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // type of face culling
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // vertex order for faces to be front-facing, can be clockwise/counter clockwise
        // depth values by adding a constant value or biasing them based on a fragments slope
        rasterizer.depthBiasEnable = VK_FALSE;
//        rasterizer.depthBiasConstantFactor = 0.0f;
//        rasterizer.depthBiasClamp = 0.0f;
//        rasterizer.depthBiasSlopeFactor = 0.0f;
        
        // Multisampling: anti-aliasing, combines fragment shader results of multiple polygons that rasterize to the same pixel
        // Mainly occurs along edges, where most noticeable aliasing artifacts occur
        // Enabling requires GPU feature
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // optional
        multisampling.pSampleMask = nullptr; // optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // optional
        multisampling.alphaToOneEnable = VK_FALSE; // optional
        
        // Color blending: once fragment shader has returned a color, needs to be combined with the color that is already in the framebuffer
        // Two ways: mix the old and new value to produce a final color OR combine the old and new value using a bitwise operation
        // VkPipelineColorBlendAttachmentState: contains the configuration per attached framebuffer
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        // If false, new color from the fragment shader is passed through unmodified
        // Otherwise, the two mixing operations are performed to compute a new color then AND with colorWriteMask
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // optional
        
        // VkPipelineColorBlendStateCreateInfo: contains the global color blending settings
        // References the array of structs for all of the framebuffers
        // Allows you to set blend constants that you can use as blend factors
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE; // If true, bitwise combo blending
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // optional
        colorBlending.blendConstants[1] = 0.0f; // optional
        colorBlending.blendConstants[2] = 0.0f; // optional
        colorBlending.blendConstants[3] = 0.0f; // optional
        
        // Limited amount of pipeline state can be changed w/o recreating the pipeline at draw time (viewport size, line width, blend constant)
        // Ignores config of these values, can specify data at drawing time
        std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        
        // Uniform values: globals (similar to dynamic state variables) can be changed at drawing time to alter the behavior of shaders w/o recreating them
        // Commonly used to pass the transformation matrix to the vertex shader, or to create texture samplers in the fragment shader
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1; // optional
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // optional
        pipelineLayoutInfo.pushConstantRangeCount = 0; // optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
        
        // Finally create graphics pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        
        // Reference array of VkPipelineShaderStageCreateInfo structs
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        
        pipelineInfo.layout = pipelineLayout; // fixed fn stage
        
        pipelineInfo.renderPass = renderPass; // renderpass reference
        pipelineInfo.subpass = 0; // sub pass index, where graphics pipeline will be used
        
        // Can create a new graphics pipeline by deriving from an existing pipeline
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // optional
        pipelineInfo.basePipelineIndex = -1; // optional
        
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        
        // Destroy shader modules
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    
    // readFile: read all bytes from file & return them in a byte array managed by std::vector
    // Open file with 2 flags: ate(start reading at end of file), binary(read the file as binary file, avoid text transforms)
    // Load the binary data from the files.
    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        
        // Read at end of file to that we use the read position to determine the size of the file and allocate a buffer
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
        
        // Seek to start of file, read all bytes at once
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        
        // Close file, return bytes
        file.close();
        return buffer;
    }
    
    // createShaderModule: take buffer with bytecode and create a VkShaderModule with it
    // Specify ptr to buffer with bytecode & length
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); // size of the bytecode is specified in bytes, bytecode pointer is a uint32_t pointer
        
        // Create shader module
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }
    
    // Set framebuffer attachments that will be used while rendering
    // Specify # of color, depth buffers + samples to use for each of them and how to handle its contents throughout rendering operations
    // Single color buffer attachment represented by one of the images from the swap chain
    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        // Format of color attachment should match format of swap chain images
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // no multisampling
        
        // loadOp, storeOp: determine what to do with attachment data before/after rendering
        // Apply to color and depth data
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // Clear the values to a constant at the start
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // Rendered contents stored in memory and can be read later
        
        // Apply to stencil data
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        
        // Images(pixels in memory) need to be transitioned to specific layouts that are suitable for the operation they will be involved in next
        // initialLayout: which layout the image will have before the render pass begins
        // finalLayout: layout to automatically transition to when the render pass finishes
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // don’t care what previous layout the image was in
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // image ready for presentation using the swap chain after rendering
        
        // Attachment: specifies attachment to reference by its index in the attachment descriptions array
        // Layout: specifies which layout the attachment will have during a subpass that uses this reference
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // Subpasses: subsequent rendering operations that depend on the contents of framebuffers in previous passes (postprocessing sequence)
        // References >= 1 of the attachments
        // Attachment index: reference from frag shader
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        
        // Use subpass dependency to ensure that the render passes don’t begin until swap chain to finish reading from the image
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        
        // Create render pass obj
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }
    
    // Framebuffer object: references all of the VkImageView objects that represent the attachments, here is color attachment
    // Image used for attachment depends on which image the swap chain returns when we retrieve one for presentation,
    // must create a framebuffer for all of the images in the swap chain and use the one that corresponds to the retrieved image at drawing time
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size()); // resize container to hold framebuffers
        
        // Iterate through the image views and create framebuffers from them
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = { swapChainImageViews[i] };
            
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass; // specify renderPass the framebuffer needs to be compatible
            
            // Specify VkImageView objects that should be bound to the respective attachment descriptions in the render pass pAttachment array
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1; // # layers in image arrays (single image, layer = 1)

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }
    
    // Need to record all of the operations you want to perform in command buffer objects
    // Command pools: manage the memory that is used to store the buffers and command buffers are allocated from them
    // Must create a command pool before we can create command buffers
    // Each command pool can only allocate command buffers that are submitted on a single type of queue
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        
        // Recording a command buffer every frame, want to be able to reset and rerecord over it
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // allow command buffers to be rerecorded individually
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // choose graphicsFamily (record commands for drawing)

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }
    
    // Command buffers: specifies the command pool and number of buffers to allocate, executed by submitting them on one of the device queues
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // specifies if the allocated command buffers are primary or secondary command buffers
        allocInfo.commandBufferCount = 1; // only allocating one command buffer
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
    
    // recordCommandBuffer: writes the commands we want to execute into a command buffer
    // takes in VkCommandBuffer, index of the current swapchain image we want to write to
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; // specifies how we’re going to use the command buffer
        beginInfo.flags = 0; // optional, parameter specifies how we’re going to use the command buffer
        beginInfo.pInheritanceInfo = nullptr; // optional, specifies which state to inherit from the calling primary command buffers
        
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        
        // Create render pass
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex]; // bind the framebuffer for the swapchain image we want to draw to, with imageIndex

        // Determing size of render area: where shader loads and stores will take place, should match size of attachments (best performance)
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        // clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}}; // black, 100% opacity
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
        
        // Start renderpass: all fn that record commands can be recognized by their vkCmd prefix
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Bind graphics pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline); // 2) specifies if the pipeline object is a graphics/compute pipeline

        // Set viewport and scissor state in command buffer b/c set as dynamic
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        
        // Bind the vertex buffer during rendering operation
        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets); // bind vertex buffers to bindings
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16); // bind index buffer to binding, only one index buffer
        
        // bind the right descriptor set for each frame to the descriptors in the shader
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0); // Draw command
        
        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
    
    // Create semaphores & fence
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }
    
    void drawFrame() {
        // 1) Wait for the previous frame to finish (waits for host to be signaled by fence)
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // 2) Acquire an image from the swap chain (GPU)
        // imageIndex specifies swap chain image to use
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        
        // Only reset the fence if we are submitting work
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        // 3) Record a command buffer which draws the scene onto that image
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        
        updateUniformBuffer(currentFrame); // update uniform buffer

        // 4) Submit the recorded command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // configure queue submission and synchronization
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        
        // specify which command buffers to actually submit for execution & semaphores to signal
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        
        // execute commands that draw acquired image (GPU)
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        
        // 5) Present the swap chain image, returning it to the swapchain (GPU)
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        
        // Set swap chains to present images to
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex; // index of the image for each swap chain
        presentInfo.pResults = nullptr; // optional, specify an array of VkResult values to check for every individual swap chain if presentation was successful
                
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // update curr frame
    }
    
    // Input: buffer size, memory properties, usage, output variables to write the handles to
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size; // size of the buffer in bytes
        bufferInfo.usage = usage; // purpose of data in buffer
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer queue family access
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }
        
        // Allocate memory for buffer, need to query its memory requirements
        // size: required amount of memory in bytes, may differ from bufferInfo.size
        // alignment: offset in bytes where the buffer begins in the allocated region of memory, depends on bufferInfo.usage, bufferInfo.flags
        // memoryTypeBits: bit field of the memory types suitable for buffer
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }
        
        // Memory allocation successful, associate this memory with the buffer
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    
    // Allocate a temporary command buffer b/c memory transfer operations use command buffers
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
        
        // Start recording the command buffer:
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // cmd buffer used once

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
       
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // optional
        copyRegion.dstOffset = 0; // optional
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion); // transfer buffer contents
       
        // Stop recording after coping
        vkEndCommandBuffer(commandBuffer);

        // Excecute transfer on buffers
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);  // wait for transfer to complete
       
        // Clean up cmd buffer
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
    
    // Vertex Buffer: memory region used for storing arbitrary data that can be read by the graphics card, store vertex data
    // Must allocate memory for the buffer
    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        
        // host visible buffer as temporary buffer
        // Staging buffer: used in CPU accessible memory to upload the data from the vertex array to
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        
        // Copy vertex data to buffer
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        // Final vertex buffer in device local memory
        // Use a buffer copy command to move the data from the staging buffer to the actual vertex buffer
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        
        // Clean up staging buffer
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    // Memory heaps: distinct memory resources (ex. dedicated VRAM and swap space in RAM for when VRAM runs out)
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        // Search for suitable memory type for buffer
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            // specify the bit field of memory types that are suitable: suitable memory type index if the corresponding bit is set to 1
            // memory must support property of our data
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties)) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }
    
    // Index buffer: array of pointers into the vertex buffer
    // Allows you to reorder the vertex data, and reuse existing data for multiple vertices
    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        // Create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        // Copy vertex data to buffer
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // Use a buffer copy command to move the data from the staging buffer to the actual vertex buffer
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        // Clean up staging buffer
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    // Descriptor: a way for shaders to freely access resources like buffers and images
    // Descriptor set layout: specifies the types of resources that are going to be accessed by the pipeline
    // Descriptor set: specifies the actual buffer or image resources that will be bound to the descriptors, bound for the drawing commands
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0; // binding used in shader
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // uniform buffer obj
        uboLayoutBinding.descriptorCount = 1; // # values in array
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // shader stage descriptor is referenced by
        uboLayoutBinding.pImmutableSamplers = nullptr; // optional, relevant for image sampling related descriptors

        // Specify the descriptor set layout during pipeline creation to tell Vulkan which descriptors the shaders will be using
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    
    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        // Need as many uniform buffers as frames in flight
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            
            // map the buffer right after creation to get a pointer to write the data later
            // persistent mapping: buffer stays mapped to this pointer for the application’s whole lifetime
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }
    
    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        // define the model, view and projection transformations in the uniform buffer object
        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        // copy the data in the uniform buffer object to the current uniform buffer
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }
    
    // Descriptor sets must be allocated from a pool
    void createDescriptorPool() {
        // describe which descriptor types our descriptor sets are going to contain, how many of them
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT); // allocate a descriptor for every frame
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT); // maximum number of descriptor sets that can be allocated

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }
    
    // Allocate descriptor set: specify descriptor pool to allocate from, the number of descriptor sets to allocate, and the descriptor set layout to base them on
    void createDescriptorSets() {
        // create one descriptor set for each frame in flight, all with the same layout
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();
        
        // allocate descriptor set
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        
        // configure each descriptor
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // specify buffer and region within it that contains the data for the descriptor
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);
            
            // how config of descriptors is updated
            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo; // descriptors that refer to buffer data
            descriptorWrite.pImageInfo = nullptr; // optional,  descriptors that refer to image data
            descriptorWrite.pTexelBufferView = nullptr; // optional, descriptors that refer to buffer views
            
            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

};

int main() {
    // Initialise the renderer

    // Enter frame by frame loop
        // Prepare frames

        // Set state on GPU

        // Present to screen

        // Check to see if still running

    // Shut down processes on renderer
    RenderApplication app;
    
    try {
        app.run();
    } catch (const std::exception& e) { // report & propogate errors
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
