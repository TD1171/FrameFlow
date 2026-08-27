cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# Target GPU architecture -- MUST be set before project().
#
# project(... CUDA) initialises CMAKE_CUDA_ARCHITECTURES itself, so testing
# `if(NOT DEFINED ...)` afterwards never fires and the setting is silently
# ignored: the build then targets CMake's default (compute_52, a 2014 Maxwell
# part) instead of this GPU. Verify with `nvcc ... --generate-code` in the
# build log if this is ever changed.
#
# 120 = compute capability 12.0 (Blackwell), confirmed on the development
# machine by cudaGetDeviceProperties rather than assumed. Override for other
# hardware:  cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(frameflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
  src/kernels.cu
  src/pipeline.cu
)

target_include_directories(frameflow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(frameflow PRIVATE ${OpenCV_LIBS})

# CUDA_SEPARABLE_COMPILATION is deliberately OFF. No .cu file calls a __device__
# function defined in another, so relocatable device code buys nothing here --
# and the device-link step it adds is broken by the space in this repository's
# path: CMake emits -Xcompiler=-Fd"...\CUDA Project\build\..." and nvcc splits
# that argument at the space, handing cl.exe a nonexistent source file.
# If kernels are ever split across translation units, either turn this on AND
# move the repo to a path without spaces, or keep all __device__ code in one .cu.
set_target_properties(frameflow PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ---------------------------------------------------------------------------
# Synchronization mode
#
# Debug builds sync after every kernel launch so a fault is attributed to the
# launch that caused it. Release builds check the launch for errors but do not
# force a sync, because a cudaDeviceSynchronize() per launch serializes the
# pipeline and would be measured as kernel time. --debug-sync forces the debug
# behaviour at runtime in any build. See include/cuda_check.h.
# ---------------------------------------------------------------------------
target_compile_definitions(frameflow PRIVATE
  $<$<CONFIG:Debug>:FRAMEFLOW_DEBUG_SYNC=1>
)

if(MSVC)
  target_compile_options(frameflow PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
  )
else()
  target_compile_options(frameflow PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  )
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
message(STATUS "FrameFlow: CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")
