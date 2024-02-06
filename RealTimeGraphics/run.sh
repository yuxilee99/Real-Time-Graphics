#build:
clang++ -g -Wall -o main main.cpp -I/opt/homebrew/include -I/Users/yuxi01px2018/VulkanSDK/1.3.275.0/macOS/include -std=c++20 -L/opt/homebrew/lib -L/Users/yuxi01px2018/VulkanSDK/1.3.275.0/macOS/lib -lglfw -lvulkan -arch arm64

#run:
VK_ICD_FILENAMES=/Users/yuxi01px2018/VulkanSDK/1.3.275.0/macOS/share/vulkan/icd.d/MoltenVK_icd.json \
VK_LAYER_PATH=/Users/yuxi01px2018/VulkanSDK/1.3.275.0/macOS/share/vulkan/explicit_layer.d \
DYLD_LIBRARY_PATH=/Users/yuxi01px2018/VulkanSDK/1.3.275.0/macOS/lib:$DYLD_LIBRARY_PATH \
./main
