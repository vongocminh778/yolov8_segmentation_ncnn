cmake_minimum_required(VERSION 3.5)
project(main)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pie -fPIE -fPIC -Wall -O3")

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()

# # NCNN
set(ncnn_DIR ./ncnn/build/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# Source files
file(GLOB SRC "*.h" "*.cpp")

add_executable(main ${SRC})
target_link_libraries(main ncnn ${OpenCV_LIBS})