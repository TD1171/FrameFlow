cmake_minimum_required(VERSION 3.24)

# Must be set before project(), which initialises this variable itself.
# 120 = Blackwell, the development GPU. Override with
# -DCMAKE_CUDA_ARCHITECTURES=<arch> for other hardware.
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

# OpenCV is used for video I/O, PNG output and the CPU baseline only. Every
# device-side operation is a custom kernel.
if(WIN32 AND NOT DEFINED OpenCV_DIR AND EXISTS "C:/opencv/build")
  set(OpenCV_DIR "C:/opencv/build")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

add_executable(frameflow
  src/main.cpp
  src/video_io.cpp
  src/cpu_baseline.cpp
  src/benchmark.cpp
  src/kernels.cu
  src/pipeline.cu
)

target_include_directories(frameflow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(frameflow PRIVATE ${OpenCV_LIBS})

# CUDA_SEPARABLE_COMPILATION stays off: no .cu file calls a __device__ function
# defined in another, so relocatable device code buys nothing here.
set_target_properties(frameflow PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Debug builds synchronize after every kernel launch so a fault is attributed
# to its own launch. Release builds do not, since that would serialize the
# pipeline and be measured as kernel time. --debug-sync forces it at runtime.
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

# On Windows the OpenCV DLL must sit beside the executable or on PATH.
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
