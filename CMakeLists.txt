# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or 
# http://www.opensource.org/licenses/mit-license.php 

project(dirichlet)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_COLOR_MAKEFILE ON)
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) # Release; debug; ...
  #set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...
ENDIF()

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
  $ENV{WORKSPACE_HOME}/workspace_research/3rdparty/include
  /usr/share/pyshared/numpy/core/include
  /usr/include
  )

# Include library directories
link_directories(
  $ENV{WORKSPACE_HOME}/workspace_research/3rdparty/lib
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

# wrapper for python
FIND_PACKAGE(Boost 1.46.1)
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
  FIND_PACKAGE(Boost 1.46.1 COMPONENTS python random)

  ADD_LIBRARY(hdp SHARED src/hdp_py.cpp)
  TARGET_LINK_LIBRARIES(hdp ${Boost_LIBRARIES} armadillo python2.7)
  INSTALL(TARGETS hdp LIBRARY DESTINATION /home/jstraub/workspace_research/bnp/python)
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()

