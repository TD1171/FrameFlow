cmake_minimum_required(VERSION 3.24)

# CUDA is added as a project language in Stage 2, together with the first
# kernel. Stage 1 is deliberately host-only so that video decode/encode can be
# proven to work before any GPU code exists to confuse a failure.
project(frameflow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# OpenCV: video decode/encode, PNG output, and the CPU baseline only.
# No GPU work goes through OpenCV -- every device-side operation in this
# project is a hand-written __global__ kernel.
# ---------------------------------------------------------------------------
if(WIN32 AND NOT DEFINED OpenCV_DIR AND EXISTS "C:/opencv/build")
  set(OpenCV_DIR "C:/opencv/build")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

add_executable(frameflow
  src/main.cpp
  src/video_io.cpp
)

target_include_directories(frameflow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(frameflow PRIVATE ${OpenCV_LIBS})

set_target_properties(frameflow PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(MSVC)
  target_compile_options(frameflow PRIVATE /W3 /utf-8)
else()
  target_compile_options(frameflow PRIVATE -Wall -Wextra)
endif()

# On Windows the OpenCV runtime DLL must sit beside the executable or on PATH.
# Copying it after the build makes the binary runnable without a PATH edit.
if(WIN32)
  foreach(_ocv_lib IN LISTS OpenCV_LIBS)
    if(TARGET ${_ocv_lib})
      add_custom_command(TARGET frameflow POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${_ocv_lib}> $<TARGET_FILE_DIR:frameflow>
        COMMAND_EXPAND_LISTS)
    endif()
  endforeach()
endif()

message(STATUS "FrameFlow: OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")
