cmake_minimum_required(VERSION 3.2)
project(face_air)

set(CMAKE_CXX_STANDARD 11)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${test_opencv_SOURCE_DIR}/bin)
# set the directory of executable files
add_executable(face_air src/main.cpp)
aux_source_directory(./src DIRSRCS)
add_executable(detect_recog_track ${DIRSRCS} include/inference.h src/inference.cpp include/sample_functions.h)
include_directories(include)

# setup CUDA
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_53,code=sm_53
        -gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -gencode arch=compute_72,code=sm_72
    )
endif()
# this derictory includes cuda related header files
include_directories(/usr/local/cuda-9.0/targets/x86_64-linux/include)

#setup OpenCV

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})

# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(face_air ${OpenCV_LIBS})
