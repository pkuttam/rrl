cmake_minimum_required(VERSION 3.16)

project(RRL)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")

# Set GCC flags
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -fopenmp -O3 -w -funroll-loops")

find_package(Eigen3 3.3.90 EXACT CONFIG REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "Found Eigen3!")
    SET(EIGEN3_INCLUDE_DIRS /home/grasping/.local/include/)
    message(STATUS "${EIGEN3_INCLUDE_DIRS}")
endif()
find_package(Tensorflow CONFIG REQUIRED)
set(program_options date_time system)
find_package(Boost COMPONENTS date_time system thread filesystem REQUIRED)
if (Boost_FOUND)
    message(STATUS "Found Boost")
    message(STATUS "${Boost_LIBRARIES}")

endif()

find_package(raiCommon CONFIG REQUIRED)
link_directories(/home/grasping/project/legged/rai_install/lib/)

find_package(raisim CONFIG REQUIRED)
find_package(raisimOgre 0.5.0 CONFIG REQUIRED)

INCLUDE_DIRECTORIES(${raiCommon_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${TensorFlow_INCLUDE_DIRS})
include_directories(include ./)
include_directories(/home/grasping/.local/include/)

add_executable(main app/main.cpp )
target_link_libraries(main ${TensorFlow_LIBRARIES} ${Boost_LIBRARIES} raisim::raisim raisim::raisimOgre)

add_executable(anymal app/train_walk.cpp)
target_link_libraries(anymal ${TensorFlow_LIBRARIES} ${Boost_LIBRARIES} raisim::raisim raisim::raisimOgre glog gflags stdc++fs ${raiCommon_LIBRARIES} ${raiSim_LIBRARIES}   ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY})