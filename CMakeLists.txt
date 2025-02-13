cmake_minimum_required(VERSION 3.10)

project(SuprasFirstPass)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

execute_process(
        COMMAND bash -c "echo $MY_LLVM_SRC"
        OUTPUT_VARIABLE MY_LLVM_SRC
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(LLVM_DIR "${MY_LLVM_SRC}/build/lib/cmake/llvm")
set(MLIR_DIR "${MY_LLVM_SRC}/build/lib/cmake/mlir")

# Find LLVM and MLIR packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)


list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")


# Add MLIR specific CMake modules
include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(tool)
add_subdirectory(lib)

