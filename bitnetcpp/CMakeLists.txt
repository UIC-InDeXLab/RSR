cmake_minimum_required(VERSION 3.10)
project("RSR.cpp")

add_subdirectory(BitNet/3rdparty/llama.cpp/ggml)

include_directories(
    BitNet/3rdparty/llama.cpp/ggml/include    
    BitNet/3rdparty/llama.cpp/ggml/src
    BitNet/include
    BitNet/src
    .
    # ../native
)

file(GLOB BITNET_SRC
    BitNet/src/*.cpp
    *.cpp
)

# list(APPEND BITNET_SRC
#     ../native/rsr.cpp
#     ../native/utils.cpp
#     ../native/naive.cpp
# )

add_executable(main main.cpp ${BITNET_SRC})

target_link_libraries(main ggml)