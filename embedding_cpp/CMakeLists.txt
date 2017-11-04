cmake_minimum_required(VERSION 3.2)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

find_package(CUDA QUIET REQUIRED)
find_package(MKL REQUIRED)

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-std=c++11 -fopenmp")

# Specify binary name and source file to build it from
cuda_add_executable(
    embedding
    utils.h
    utils.cc
    utils.cu
    solver.cu
    solver.cc
    solver.h
    cxxopts.h
    embedding.cc
)

target_link_libraries(embedding
    mkl_intel_lp64
    mkl_sequential
    mkl_core
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cusparse_LIBRARY})
