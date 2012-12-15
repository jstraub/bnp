project(dirichlet)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release) # Release; debug; ...
#set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...

# add flags depending on build type
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -O2 -finline-functions -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# add flags in general
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_EXE_LINKER_FLAGS "")


IF(DEFINED ENV{WORKSPACE_HOME})
  message("...using WORKSPACE_HOME found in $ENV{WORKSPACE_HOME}")
ELSE()
  message("WORKSPACE_HOME is not defined.  You must tell CMake where to find WORKSPACE_HOME")
  # exit early 
  return()
ENDIF()

# Include directories (.h .hpp files)
include_directories(
  ./include
  #$ENV{WORKSPACE_HOME}/workspace_research/3rdparty/armadillo-3.4.4/include
  #$ENV{WORKSPACE_HOME}/workspace_research/3rdparty/include
  /usr/include
  )

# Include library directories
link_directories(
  #$ENV{WORKSPACE_HOME}/workspace_research/3rdparty/armadillo-3.4.4/
  #$ENV{WORKSPACE_HOME}/workspace_research/3rdparty/lib
  /usr/lib/
  )

# define the sources
set(SRC
 src/hdp.cpp
  )

set(LIBS
 armadillo
 boost_random
 )

# add executable that should be compiled
add_executable(testHdp ./src/testHdp.cpp)
target_link_libraries(testHdp ${LIBS} stdc++)

add_executable(hdpCluster ./src/hdpCluster.cpp)
target_link_libraries(hdpCluster ${LIBS} stdc++)

add_executable(testRandom ./src/testRandom.cpp)
target_link_libraries(testRandom ${LIBS} stdc++)

add_executable(testArmadillo ./src/testArmadillo.cpp)
target_link_libraries(testArmadillo ${LIBS} stdc++)
