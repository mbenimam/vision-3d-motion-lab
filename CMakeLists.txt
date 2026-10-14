cmake_minimum_required(VERSION 3.24)
project(vision_lab VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VL_BUILD_APP "Build the Qt application" ON)
option(VL_BUILD_TESTS "Build the unit tests" ON)

# Modules were renamed in OpenCV 5 (calib3d -> geometry, features2d -> features). Link only the
# modules we use, under whichever name is installed; src/core/common/cv_compat.hpp does the same
# for headers.
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS 4.5)
  message(FATAL_ERROR "OpenCV 4.5 or newer is required (found ${OpenCV_VERSION})")
endif()
set(VL_CV_LIBS)
foreach(m core imgproc imgcodecs videoio video features features2d geometry calib3d)
  if(TARGET opencv_${m})
    list(APPEND VL_CV_LIBS opencv_${m})
  endif()
endforeach()
message(STATUS "OpenCV ${OpenCV_VERSION}: ${VL_CV_LIBS}")

add_library(vl_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(vl_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -Wno-missing-field-initializers)
endif()

# ---------------------------------------------------------------- core: C++ and OpenCV only, no Qt
# One static library per area, so each area builds and tests on its own.
function(vl_core_area name)
  file(GLOB_RECURSE srcs CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/core/${name}/*.cpp)
  if(NOT srcs)  # an area with headers only (or not written yet) still gets a target
    set(srcs ${CMAKE_SOURCE_DIR}/src/core/common/empty.cpp)
  endif()
  add_library(vl_${name} STATIC ${srcs})
  target_include_directories(vl_${name} PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(vl_${name} PUBLIC ${VL_CV_LIBS} PRIVATE vl_warnings ${ARGN})
endfunction()

vl_core_area(common)
foreach(area geometry stereo motion flow)
  vl_core_area(${area} vl_common)
  target_link_libraries(vl_${area} PUBLIC vl_common)
endforeach()

# ---------------------------------------------------------------- app: Qt 6 Widgets, model / controller / view
if(VL_BUILD_APP)
  find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTORCC ON)

  file(GLOB fw_srcs CONFIGURE_DEPENDS
       src/app/model/*.cpp src/app/controller/*.cpp src/app/view/*.cpp src/app/theme/*.cpp src/app/*.cpp)
  list(REMOVE_ITEM fw_srcs ${CMAKE_SOURCE_DIR}/src/app/main.cpp)
  add_library(vl_framework STATIC ${fw_srcs} src/app/theme/theme.qrc)
  target_include_directories(vl_framework PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(vl_framework PUBLIC Qt6::Widgets Qt6::Concurrent vl_common PRIVATE vl_warnings)
  target_compile_definitions(vl_framework PRIVATE VL_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

  # Modules register themselves, so they are object libraries linked whole into the executable.
  set(module_objs)
  foreach(area geometry stereo motion flow)
    file(GLOB msrcs CONFIGURE_DEPENDS src/app/modules/${area}/*.cpp)
    if(msrcs)
      add_library(vl_modules_${area} OBJECT ${msrcs})
      target_link_libraries(vl_modules_${area} PUBLIC vl_framework vl_${area} PRIVATE vl_warnings)
      list(APPEND module_objs $<TARGET_OBJECTS:vl_modules_${area}>)
      list(APPEND module_libs vl_${area})
    endif()
  endforeach()

  add_executable(vision-lab src/app/main.cpp ${module_objs})
  target_link_libraries(vision-lab PRIVATE vl_framework ${module_libs} vl_warnings)
endif()

# ---------------------------------------------------------------- tests
if(VL_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)
  foreach(area common geometry stereo motion flow)
    file(GLOB tsrcs CONFIGURE_DEPENDS tests/${area}/*.cpp)
    if(tsrcs)
      add_executable(test_${area} ${tsrcs})
      target_link_libraries(test_${area} PRIVATE vl_${area} GTest::gtest_main vl_warnings)
      gtest_discover_tests(test_${area} DISCOVERY_TIMEOUT 60)
    endif()
  endforeach()
endif()
