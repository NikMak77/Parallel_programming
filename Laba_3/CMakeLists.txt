cmake_minimum_required(VERSION 3.12)

project("Laba_3_MPI" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matrix_mul
  "matrix_mul.cpp"
)

target_link_libraries(matrix_mul PRIVATE MPI::MPI_CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(matrix_mul PRIVATE "-Wall" "-Weffc++" "-Wextra" "-Wpedantic" "-O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif()
  target_compile_options(matrix_mul PRIVATE "/W4" "/utf-8" "/O2")
endif()