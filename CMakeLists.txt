# Distributed under the MIT License (See accompanying file /LICENSE)
# CMake build : tweedledum library
cmake_minimum_required(VERSION 3.20...3.31)
project(tweedledum VERSION 1.2.0 LANGUAGES C CXX)

# C++ standard: C++17 is the minimum; you may pass -DCMAKE_CXX_STANDARD=20.
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Check if tweedledum is being used directly or via add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND NOT DEFINED TWEEDLEDUM_MASTER_PROJECT)
    set(TWEEDLEDUM_MASTER_PROJECT ON)
else()
    set(TWEEDLEDUM_MASTER_PROJECT OFF)
endif()

# Options
# =============================================================================
# When built through pip (scikit-build-core sets SKBUILD) we only want the
# Python extension; for plain CMake builds the bindings are opt-in.
if(SKBUILD)
    set(_tweedledum_pybinds_default ON)
else()
    set(_tweedledum_pybinds_default OFF)
endif()
option(TWEEDLEDUM_PYBINDS "Build python bindings" ${_tweedledum_pybinds_default})
option(TWEEDLEDUM_TESTS "Build C++ tests" OFF)
option(TWEEDLEDUM_EXAMPLES "Build C++ examples" OFF)
option(TWEEDLEDUM_ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)
option(TWEEDLEDUM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(ENABLE_BILL_Z3 "Enable Z3 interface for bill library" OFF)
option(TWEEDLEDUM_STATIC_LIBSTDCXX
    "Statically link libstdc++ into the Python module (Linux/GCC)" ON)

# The library is linked into a Python extension module, so it must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 3rd-party libraries (all vendored under external/)
# =============================================================================
add_subdirectory(external)

# pybind11 (only needed for the Python bindings)
# When building through pip, scikit-build-core makes the `pybind11` package from
# the build requirements discoverable.  For plain CMake builds we fall back to
# the pybind11 installed in the active Python environment, or download it.
if(TWEEDLEDUM_PYBINDS AND TWEEDLEDUM_MASTER_PROJECT AND NOT TARGET pybind11::module)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    find_package(pybind11 2.13 CONFIG QUIET HINTS "${_pybind11_dir}")
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found in the Python environment; downloading it")
        include(FetchContent)
        FetchContent_Declare(pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz)
        FetchContent_MakeAvailable(pybind11)
    else()
        message(STATUS "Using pybind11 ${pybind11_VERSION} from ${pybind11_DIR}")
    endif()
endif()

# Warnings
# =============================================================================
add_library(tweedledum_warnings INTERFACE)
target_compile_options(tweedledum_warnings INTERFACE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wextra
        $<$<BOOL:${TWEEDLEDUM_WARNINGS_AS_ERRORS}>:-Werror>>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /utf-8 /bigobj
        $<$<BOOL:${TWEEDLEDUM_WARNINGS_AS_ERRORS}>:/WX>>)

# Library
# =============================================================================
add_library(tweedledum STATIC)
add_library(tweedledum::tweedledum ALIAS tweedledum)
target_include_directories(tweedledum PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(tweedledum
    PUBLIC
        Eigen3::Eigen3
        fmt::fmt-header-only
        mockturtle
        nlohmann_json
    PRIVATE
        tweedledum_warnings)
if(TWEEDLEDUM_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tweedledum PUBLIC -O0 -g --coverage)
    target_link_options(tweedledum PUBLIC --coverage)
endif()
add_subdirectory(src)

# Python bindings
# =============================================================================
if(TWEEDLEDUM_PYBINDS AND TWEEDLEDUM_MASTER_PROJECT)
    pybind11_add_module(_tweedledum MODULE)
    target_include_directories(_tweedledum PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(_tweedledum PRIVATE tweedledum tweedledum_warnings)
    # Python interpreters from conda (and some other distributions) ship their
    # own, often older, libstdc++ and load it before the system one.  A module
    # compiled with a newer GCC then fails to import with
    # "GLIBCXX_3.4.xx not found".  Linking the C++ runtime statically makes the
    # module independent of whichever libstdc++ the interpreter loaded.
    if(TWEEDLEDUM_STATIC_LIBSTDCXX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_LINK_OPTIONS -static-libstdc++ -static-libgcc)
        check_cxx_source_compiles(
            "#include <string>\nint main() { std::string s(\"x\"); return (int) s.size() - 1; }"
            TWEEDLEDUM_HAS_STATIC_LIBSTDCXX)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
        if(TWEEDLEDUM_HAS_STATIC_LIBSTDCXX)
            target_link_options(_tweedledum PRIVATE -static-libstdc++ -static-libgcc)
        else()
            message(WARNING
                "Static libstdc++ is not available (on Fedora/RHEL install "
                "'libstdc++-static'). The Python module will link libstdc++ "
                "dynamically; if importing it fails with 'GLIBCXX_... not "
                "found' see the Troubleshooting section of README.md.")
        endif()
    endif()
    add_subdirectory(python/tweedledum)
    if(SKBUILD)
        install(TARGETS _tweedledum LIBRARY DESTINATION tweedledum)
    endif()
endif()

# Install (C++ library only; the Python package is installed through pip)
# =============================================================================
if(NOT SKBUILD)
    include(GNUInstallDirs)
    install(TARGETS tweedledum
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# Examples
# =============================================================================
if(TWEEDLEDUM_EXAMPLES AND NOT SKBUILD)
    add_subdirectory(examples)
endif()

# Tests
# =============================================================================
if(TWEEDLEDUM_MASTER_PROJECT AND TWEEDLEDUM_TESTS AND NOT SKBUILD)
    enable_testing()
    add_subdirectory(tests)
endif()
