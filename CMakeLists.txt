cmake_minimum_required(VERSION 3.12)
project(fast_lio)

# RelWithDebInfo
SET(CMAKE_BUILD_TYPE "Debug")

ADD_COMPILE_OPTIONS(-std=c++17 )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

# generate compile_commands.json for VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP: 4")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

# ROS compatibility
set(ROS_VERSION_ENV "$ENV{ROS_VERSION}")
if(ROS_VERSION_ENV STREQUAL "1")
  if(DEFINED ENV{AMENT_PREFIX_PATH} AND NOT "$ENV{AMENT_PREFIX_PATH}" STREQUAL "")
    message(FATAL_ERROR "Conflicting ROS environment detected: ROS_VERSION=1 but AMENT_PREFIX_PATH is set.")
  endif()
  message("\n${BoldGreen}ROS1 environment detected...${ColourReset}\n")
  add_compile_definitions(USE_ROS1)
elseif(ROS_VERSION_ENV STREQUAL "2")
  if(DEFINED ENV{CATKIN_ENV_HOOK_WORKSPACE} AND NOT "$ENV{CATKIN_ENV_HOOK_WORKSPACE}" STREQUAL "")
    message(FATAL_ERROR "Conflicting ROS environment detected: ROS_VERSION=2 but CATKIN_ENV_HOOK_WORKSPACE is set.")
  endif()
  message("\n${BoldGreen}ROS2 environment detected...${ColourReset}\n")
  add_compile_definitions(USE_ROS2)
else()
  message(FATAL_ERROR "Invalid or missing ROS_VERSION environment variable. Please source a ROS1 or ROS2 environment before configuring.")
endif()

# ROS2 CMakelist
if(ROS_VERSION_ENV STREQUAL "2")

message("\n${BoldGreen}Building for ROS2...${ColourReset}\n")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(yaml-cpp REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include
)

add_executable(fastlio_mapping
  src/laserMapping.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
  src/IMU_Processing.cpp
  src/main.cpp
)

ament_target_dependencies(fastlio_mapping
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  pcl_ros
  livox_ros_driver2
)

target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
  ${PYTHON_LIBRARIES}
  yaml-cpp
)
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})

install(
  TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

endif()

if(ROS_VERSION_ENV STREQUAL "1")

message("\n${BoldGreen}Building for ROS1...${ColourReset}\n")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver2
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(yaml-cpp REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_executable(fastlio_mapping
  src/laserMapping.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
  src/IMU_Processing.cpp
  src/main.cpp
)
target_link_libraries(fastlio_mapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${PYTHON_LIBRARIES}
  yaml-cpp
)
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
add_dependencies(fastlio_mapping ${catkin_EXPORTED_TARGETS})

endif()
