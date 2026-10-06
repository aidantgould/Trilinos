cmake_minimum_required(VERSION 3.27)
project(HookDiag CXX)
set(CMAKE_CXX_STANDARD 20)

# Trilinos from the sibling build tree, as in trilinos-teko-pyfront's
# CMakeLists.txt. This file sits five levels below <ROOT>.
get_filename_component(_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(_TRIL ${_ROOT}/trilinos-build/cmake_packages)
find_package(Teko        REQUIRED PATHS ${_TRIL}/Teko        NO_DEFAULT_PATH)
find_package(Stratimikos REQUIRED PATHS ${_TRIL}/Stratimikos NO_DEFAULT_PATH)
find_package(Ifpack2     REQUIRED PATHS ${_TRIL}/Ifpack2     NO_DEFAULT_PATH)
find_package(MPI REQUIRED CXX)

add_executable(hook_diag hook_diag.cpp)
target_link_libraries(hook_diag PRIVATE
    Teko::teko Stratimikos::stratimikos Ifpack2::ifpack2 MPI::MPI_CXX)
target_compile_options(hook_diag PRIVATE -Wno-deprecated-declarations)
