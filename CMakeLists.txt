cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fgprs LANGUAGES CXX CUDA)
# project(fgprs)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_HOME "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc")
set(CUDNN_LIBRARY_PATH "/home/amir/anaconda3/envs/libtorch/lib/libcudnn.so")
set(CUDNN_INCLUDE_PATH "/home/amir/anaconda3/envs/libtorch/include")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
# set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

set(FGPRS_INCLUDE_PATH src)

find_package(Torch REQUIRED)

add_executable(fgprs main.cpp)
find_package(fmt REQUIRED)
target_link_libraries(fgprs fmt::fmt)
add_subdirectory(${FGPRS_INCLUDE_PATH})
target_link_libraries(fgprs ${TORCH_LIBRARIES})
set_property(TARGET fgprs PROPERTY CXX_STANDARD 20)