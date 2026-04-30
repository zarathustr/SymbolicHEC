cmake_minimum_required(VERSION 3.18)
project(AXYB_CMake_CPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AXYB_USE_BUNDLED_TBB "Build and use bundled oneTBB-2019_U9" ON)
option(AXYB_ENABLE_EIGEN "Enable optional Eigen matrix backends when Eigen3 is available" ON)
set(AXYB_TBB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/oneTBB-2019_U9" CACHE PATH "Path to oneTBB-2019_U9")
set(AXYB_TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/templates" CACHE PATH "Directory containing generated AXYB .tpl files")

# Requested project-wide optimization/suppression flags.
add_compile_options(-w -O3)

if(APPLE)
    if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "macOS target architecture" FORCE)
    endif()
    if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
    endif()
endif()

# BLAS/LAPACK backend.  On Apple Silicon this resolves to Accelerate.
if(APPLE)
    set(AXYB_LAPACK_LIBS "-framework Accelerate")
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(AXYB_LAPACK_LIBS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

# Optional Eigen interfaces.  They are enabled only when Eigen3 is discoverable.
set(AXYB_EIGEN_FOUND OFF)
if(AXYB_ENABLE_EIGEN)
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        set(AXYB_EIGEN_FOUND ON)
    else()
        message(STATUS "Eigen3 was not found; Eigen-named backends will report a runtime error. BLAS/LAPACK/TBB backends remain available.")
    endif()
endif()

# Bundled oneTBB-2019_U9 build.  The explicit -j2 avoids oversubscribing small build sandboxes.
if(AXYB_USE_BUNDLED_TBB AND EXISTS "${AXYB_TBB_ROOT}/cmake/TBBBuild.cmake")
    include("${AXYB_TBB_ROOT}/cmake/TBBBuild.cmake")
    set(_tbb_cplus_flags "-w -O3")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # GCC 14 diagnoses a name-change pattern inside oneTBB-2019 as an error unless relaxed.
        set(_tbb_cplus_flags "${_tbb_cplus_flags} -fpermissive -Wno-changes-meaning")
    endif()
    set(_tbb_cxx_driver "${CMAKE_CXX_COMPILER}")
    set(_tbb_make_args -j2 stdver=c++14 tbb "CPLUS=${_tbb_cxx_driver}" "CPLUS_FLAGS=${_tbb_cplus_flags}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND _tbb_make_args compiler=clang stdlib=libc++)
    endif()
    if(APPLE)
        if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
            list(APPEND _tbb_make_args arch=arm64)
        elseif(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
            list(APPEND _tbb_make_args arch=intel64)
        endif()
        set(_axyb_macos_deployment_target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
        if(_axyb_macos_deployment_target STREQUAL "")
            set(_axyb_macos_deployment_target "11.0")
        endif()
        if(CMAKE_OSX_SYSROOT)
            list(APPEND _tbb_make_args SDKROOT=${CMAKE_OSX_SYSROOT})
        endif()
        list(APPEND _tbb_make_args MACOSX_DEPLOYMENT_TARGET=${_axyb_macos_deployment_target})
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        list(APPEND _tbb_make_args arch=arm64)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
        list(APPEND _tbb_make_args arch=intel64)
    endif()
    tbb_build(TBB_ROOT "${AXYB_TBB_ROOT}" CONFIG_DIR AXYB_TBB_CONFIG_DIR MAKE_ARGS ${_tbb_make_args})
    if(AXYB_TBB_CONFIG_DIR)
        list(PREPEND CMAKE_PREFIX_PATH "${AXYB_TBB_CONFIG_DIR}")
    endif()
endif()

find_package(TBB REQUIRED COMPONENTS tbb)

add_library(axyb_core
    src/matrix.cpp
    src/template_data.cpp
    src/linear_solver.cpp
    src/axyb_solver.cpp
    src/measurement_io.cpp
    src/generated_symbolics.cpp
)

target_include_directories(axyb_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(axyb_core PUBLIC AXYB_ENABLE_EIGEN=$<BOOL:${AXYB_EIGEN_FOUND}>)
target_link_libraries(axyb_core PUBLIC TBB::tbb PRIVATE ${AXYB_LAPACK_LIBS})
if(AXYB_EIGEN_FOUND)
    target_link_libraries(axyb_core PUBLIC Eigen3::Eigen)
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    target_compile_options(axyb_core PRIVATE -mcpu=apple-m1)
endif()

add_executable(test_AXYB_grobner_solver src/main_single.cpp)
target_link_libraries(test_AXYB_grobner_solver PRIVATE axyb_core)

add_executable(test_AXYB_grobner_solver_loop src/main_loop.cpp)
target_link_libraries(test_AXYB_grobner_solver_loop PRIVATE axyb_core)

add_executable(benchmark_AXYB_linear_backends src/main_benchmark_linear.cpp)
target_link_libraries(benchmark_AXYB_linear_backends PRIVATE axyb_core)

add_executable(run_rw_multi_eye_multi_hand src/main_rw_multi_eye_multi_hand.cpp)
target_link_libraries(run_rw_multi_eye_multi_hand PRIVATE axyb_core)

add_executable(run_AXYB_input_file_example src/main_input_file_example.cpp)
target_link_libraries(run_AXYB_input_file_example PRIVATE axyb_core)

foreach(_exe test_AXYB_grobner_solver test_AXYB_grobner_solver_loop benchmark_AXYB_linear_backends run_rw_multi_eye_multi_hand run_AXYB_input_file_example)
    add_custom_command(TARGET ${_exe} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${_exe}>/templates"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${AXYB_TEMPLATE_DIR}" "$<TARGET_FILE_DIR:${_exe}>/templates"
    )
endforeach()

install(TARGETS axyb_core test_AXYB_grobner_solver test_AXYB_grobner_solver_loop benchmark_AXYB_linear_backends run_rw_multi_eye_multi_hand run_AXYB_input_file_example
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY data/templates/ DESTINATION share/axyb/templates)
