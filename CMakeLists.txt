cmake_minimum_required(VERSION 3.0.0)
project(traffic-sign-detection VERSION 0.1.0)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(traffic-sign-detection main.cpp)

target_link_libraries(traffic-sign-detection ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
