cmake_minimum_required(VERSION 3.15)
project(OCR_Runtime)

set(CMAKE_CXX_STANDARD 17)

# Find ONNX Runtime
find_package(onnxruntime REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
file(GLOB SOURCES "src/*.cpp")

# Executable
add_executable(ocr_runtime ${SOURCES})

# Link libraries
target_link_libraries(ocr_runtime 
    ${OpenCV_LIBS}
    onnxruntime
)
