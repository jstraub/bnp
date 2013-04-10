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

FIND_PACKAGE(PythonLibs REQUIRED)
message("Found python library: ${PYTHON_LIBRARY}")
message("Found python include: ${PYTHON_INCLUDE_DIRS}")
IF( "${PYTHON_INCLUDE_DIRS}" STREQUAL "/usr/include/python2.6" )
  message("  Found python version 2.6")
  SET(PYTHON_V "PYTHON_2_6")
ENDIF()
IF( "${PYTHON_INCLUDE_DIRS}" STREQUAL "/usr/include/python2.7" )
  message("  Found python version 2.7")
  SET(PYTHON_V "PYTHON_2_7")
ENDIF()

# add flags depending on build type
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -O2 -finline-functions -DNDEBUG -fopenmp -D${PYTHON_V}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -D${PYTHON_V}")

# add flags in general
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_EXE_LINKER_FLAGS "")

IF(DEFINED ENV{WORKSPACE_HOME})
  message("--- using WORKSPACE_HOME found in $ENV{WORKSPACE_HOME}")
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
  ${PYTON_INCLUDE_DIRS}
  /usr/include
  )

# Include library directories
link_directories(
  $ENV{WORKSPACE_HOME}/workspace_research/3rdparty/lib
  /usr/lib/
  )

# define the sources
set(SRC
  src/probabilityHelpers.cpp
  )

set(LIBS
 armadillo
 boost_random
 )

# add executable that should be compiled
add_executable(testHdp ./src/testHdp.cpp ${SRC})
target_link_libraries(testHdp ${LIBS} stdc++)

add_executable(hdpCluster ./src/hdpCluster.cpp ${SRC}) 
target_link_libraries(hdpCluster ${LIBS} stdc++)

add_executable(testRandom ./src/testRandom.cpp)
target_link_libraries(testRandom ${LIBS} stdc++)

add_executable(testArmadillo ./src/testArmadillo.cpp)
target_link_libraries(testArmadillo ${LIBS} stdc++)

ADD_LIBRARY(bnp SHARED src/hdp_py.cpp ${SRC})
TARGET_LINK_LIBRARIES(bnp ${LIBS} boost_python  python2.7)
INSTALL(TARGETS bnp LIBRARY DESTINATION /home/jstraub/workspace_research/bnp/python)

