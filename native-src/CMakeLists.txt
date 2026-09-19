cmake_minimum_required(VERSION 3.22.1)
project(llama_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(llama_bridge SHARED
    llama_bridge.cpp
)

target_include_directories(llama_bridge PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
)

target_link_libraries(llama_bridge
    llama
    android
    log
)
