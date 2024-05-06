cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(main CXX)

# Find Package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS program_options date_time)


# Create Executable File
add_executable(${PROJECT_NAME} src/main.cpp
    src/step05/em.h src/step05/em.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE src)
option(ENABLE_UNIT_TESTS "Enable unit testing" OFF)
if(ENABLE_UNIT_TESTS)
    add_definitions(-DUNIT_TEST)
    find_package(Boost 1.71.0 REQUIRED COMPONENTS unit_test_framework)
    target_link_libraries(main Boost::unit_test_framework)
endif()

include_directories(
    "${PROJECT_SOURCE_DIR}"
    "${Boost_INCLUDE_DIRS}"
    "${OpenCV_INCLUDE_DIRS}"
    /usr/local/cpplinq)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} Boost::program_options Boost::date_time ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
