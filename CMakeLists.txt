cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

project(downsampler)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  nodelet
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  ecl_geometry
)

generate_dynamic_reconfigure_options(
    cfg/Downsampler.cfg
    cfg/DownsamplerRamp.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    dynamic_reconfigure
    nodelet
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    ecl_geometry
)

include_directories( include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
   src/downsampler.cpp
)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

install(DIRECTORY cfg DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(FILES nodelet_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
