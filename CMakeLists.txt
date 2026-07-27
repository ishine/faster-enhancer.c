cmake_minimum_required(VERSION 3.20)
project(fe LANGUAGES C VERSION 0.1.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FE_ENABLE_PROFILE     "Per-op timing instrumentation"            OFF)
option(FE_USE_ACCURATE_POLY  "Use Remez polys in shared 128-bit/NEON transcendentals" OFF)
option(FE_BUILD_TESTS        "Build test and benchmark runners"          ON)
include(CTest)

# Target arch detection. Supported deployment matrix is arm64 + x86_64
# (covering Apple Silicon, modern Android/iOS, MinGW or Clang GNU-driver
# Windows, Linux desktop and server). The engine ships no scalar runtime tier — see
# qgemm_dispatch.c.
if(MSVC)
    message(FATAL_ERROR
        "fe currently requires GCC/Clang-style target attributes and per-file "
        "ISA flags. Use a GNU-style Clang or MinGW toolchain on Windows; "
        "MSVC/clang-cl is not supported.")
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)
if(_proc MATCHES "aarch64|arm64|armv8")
    set(FE_ARCH "arm64")
elseif(_proc MATCHES "x86_64|amd64|x64")
    set(FE_ARCH "x86_64")
else()
    set(FE_ARCH "unknown")
endif()
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _osx_n)
    if(_osx_n EQUAL 1)
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            set(FE_ARCH "arm64")
        elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            set(FE_ARCH "x86_64")
        endif()
    endif()
endif()

if(NOT (FE_ARCH STREQUAL "arm64" OR FE_ARCH STREQUAL "x86_64"))
    message(FATAL_ERROR
        "fe supports arm64 and x86_64 only. "
        "Detected CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}'.")
endif()

message(STATUS "fe ${PROJECT_VERSION}")
message(STATUS "  arch     : ${FE_ARCH}")
message(STATUS "  dispatch : runtime (cpu-feature detection at fe_init)")

# Baseline flags. Per-tier -march flags are applied per-source below.
set(FE_FLAGS_COMMON
    -O3 -Wall -Wextra -Wno-unused-parameter
    -fno-math-errno -fno-trapping-math -ffinite-math-only
    -fmerge-all-constants -fno-exceptions
    -fno-stack-protector -fvisibility=hidden
    -DNDEBUG)
# Intentionally NOT setting -ffast-math. The runtime uses fixed SIMD
# polynomial approximations and a finite-input contract; cross-tier byte-id is
# guarded by the regression tests rather than relying on libm equivalence.
# -fno-stack-protector: kernels are leaf-ish hot paths; canary load+check
#   per call is dead weight (also we just sized down the big stack arrays
#   that would have triggered the canary heuristic anyway).
# -fvisibility=hidden: only the fe_* public ABI is annotated; internal
#   symbols stay TU-local for tighter codegen + fewer GOT lookups.
#
# LTO: opt-in (-DFE_ENABLE_LTO=ON). Measured neutral-to-slightly-negative
# on the streaming GRU/attention hot path on M2: the function-pointer
# dispatch (fe_qgemm_ops.*) is already collapsed to a direct call per tier
# inside the engine, so there's no leftover indirect call for LTO to
# devirtualize, and the resulting cross-TU inlining sometimes regresses
# register allocation in the hot I8MM kernels.
option(FE_ENABLE_LTO "Enable interprocedural optimization (LTO)" OFF)

# PGO: two-stage build. Pass 1 emits instrumentation
# (-DFE_PGO=generate, build, run with representative workload to produce
# default.profraw). Pass 2 consumes the merged profile
# (-DFE_PGO=use -DFE_PGO_PROFILE=/abs/path/to/merged.profdata).
set(FE_PGO "" CACHE STRING "PGO mode: '' | generate | use")
set(FE_PGO_PROFILE "" CACHE PATH "Merged .profdata for FE_PGO=use")
if(FE_PGO STREQUAL "generate")
    set(FE_PGO_FLAGS -fprofile-generate=${CMAKE_BINARY_DIR}/pgo-raw)
    list(APPEND FE_FLAGS_COMMON ${FE_PGO_FLAGS})
    link_libraries(${FE_PGO_FLAGS})
    message(STATUS "  PGO      : generate (raw → ${CMAKE_BINARY_DIR}/pgo-raw)")
elseif(FE_PGO STREQUAL "use")
    if(NOT EXISTS "${FE_PGO_PROFILE}")
        message(FATAL_ERROR "FE_PGO=use requires -DFE_PGO_PROFILE=<merged.profdata> (got '${FE_PGO_PROFILE}')")
    endif()
    list(APPEND FE_FLAGS_COMMON -fprofile-use=${FE_PGO_PROFILE} -Wno-profile-instr-out-of-date -Wno-profile-instr-unprofiled)
    message(STATUS "  PGO      : use (${FE_PGO_PROFILE})")
endif()
if(FE_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT FE_LTO_OK OUTPUT FE_LTO_ERR)
    if(FE_LTO_OK)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "  LTO      : enabled")
    else()
        message(STATUS "  LTO      : requested but unavailable (${FE_LTO_ERR})")
    endif()
else()
    message(STATUS "  LTO      : off (FE_ENABLE_LTO=ON to enable)")
endif()

if(FE_ENABLE_PROFILE)
    add_compile_definitions(FE_ENABLE_PROFILE)
endif()
if(FE_USE_ACCURATE_POLY)
    add_compile_definitions(FE_USE_ACCURATE_POLY)
endif()

# Engine library. The arch dispatcher always exists; per-arch kernel
# directories only contribute on the matching host.
file(GLOB FE_SRC_TOP CONFIGURE_DEPENDS src/*.c)
file(GLOB FE_SRC_NN CONFIGURE_DEPENDS src/nn/*.c)
file(GLOB FE_SRC_QGEMM CONFIGURE_DEPENDS src/qgemm/*.c)
file(GLOB FE_SRC_WINO CONFIGURE_DEPENDS src/winograd/*.c)
# FFT: per-arch tier TUs only (added below). No scalar runtime tier —
# unsupported architectures fail at configure time; x86 hosts below the
# AVX2+FMA3+F16C runtime floor fail at fe_init.
set(FE_SRC ${FE_SRC_TOP} ${FE_SRC_NN} ${FE_SRC_QGEMM} ${FE_SRC_WINO})
if(FE_ARCH STREQUAL "arm64")
    file(GLOB FE_SRC_ARM CONFIGURE_DEPENDS src/qgemm/arm/*.c)
    list(APPEND FE_SRC ${FE_SRC_ARM}
                       src/fft/fft_neon.c)
elseif(FE_ARCH STREQUAL "x86_64")
    file(GLOB FE_SRC_X86 CONFIGURE_DEPENDS src/qgemm/x86/*.c)
    list(APPEND FE_SRC ${FE_SRC_X86}
                       src/fft/fft_avx2.c
                       src/fft/fft_avx512.c)
endif()

add_library(fe STATIC ${FE_SRC})
target_include_directories(fe
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/internal)
if(NOT MSVC AND NOT WIN32)
    target_link_libraries(fe PUBLIC m)
endif()
target_compile_options(fe PRIVATE ${FE_FLAGS_COMMON})

# Per-file -march flags for each ISA tier. The CPU-detection and dispatcher
# TUs stay at baseline so they cannot SIGILL on hardware that lacks the
# very feature they're trying to probe.
if(FE_ARCH STREQUAL "arm64")
    set_source_files_properties(src/qgemm/arm/qgemm_dotprod.c PROPERTIES
        COMPILE_FLAGS "-march=armv8.2-a+dotprod")
    set_source_files_properties(src/qgemm/arm/qgemm_i8mm.c PROPERTIES
        COMPILE_FLAGS "-march=armv8.6-a+dotprod+i8mm")
elseif(FE_ARCH STREQUAL "x86_64")
    set_source_files_properties(src/qgemm/x86/qgemm_avx2.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma")
    set_source_files_properties(src/fft/fft_avx2.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma")
    set_source_files_properties(src/qgemm/x86/qgemm_avxvnni.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mavxvnni")
    set_source_files_properties(src/qgemm/x86/qgemm_avx512vnni.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mavxvnni -mavx512f -mavx512bw -mavx512vl -mavx512vnni")
    set_source_files_properties(src/fft/fft_avx512.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mavxvnni -mavx512f -mavx512bw -mavx512vl -mavx512vnni")
    # Every TU that gates SIMD via FE_QGEMM_HAVE_AVX2 must see -mavx2 so the
    # preprocessor enables the AVX2 paths. Without these flags those blocks
    # are silently preprocessed-out and the engine falls back to narrower
    # code inside otherwise-vectorized routines. F16C+FMA are needed for the
    # fp16 cvt + FMA intrinsics used in the fp16-storage paths.
    #
    # When adding a new TU that uses _mm256_* / _mm_cvtph_ps / etc, add it
    # to this list — otherwise its SIMD code blocks are silently dropped at
    # preprocess time (the latent bug originally hit on fe_winograd.c).
    set(FE_X86_AVX2_TUS
        src/winograd/fe_winograd.c
        src/qgemm/qgemm_quant.c
        src/nn/fe_activations.c
        src/nn/fe_attention.c
        src/fe_engine.c
        src/nn/fe_gru.c
        src/nn/fe_qgemm.c
        src/nn/fe_sgemm.c
        src/fe_stft.c
        src/nn/fe_vec.c)
    set_source_files_properties(${FE_X86_AVX2_TUS} PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mf16c")
endif()

# Test runners (fe_run_file: end-to-end inference; fe_test_cross_tier: bit-id).
if(FE_BUILD_TESTS)
    file(GLOB TEST_SRC CONFIGURE_DEPENDS tests/*.c)
    foreach(T ${TEST_SRC})
        get_filename_component(TNAME ${T} NAME_WE)
        add_executable(${TNAME} ${T})
        target_include_directories(${TNAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/internal)
        target_link_libraries(${TNAME} PRIVATE fe)
        target_compile_options(${TNAME} PRIVATE ${FE_FLAGS_COMMON})
    endforeach()

    if(BUILD_TESTING)
        set(FE_TEST_WEIGHTS "${CMAKE_CURRENT_SOURCE_DIR}/weights/fe.q8")
        set(FE_TEST_AUDIO   "${CMAKE_CURRENT_SOURCE_DIR}/testaudio/street_5dB.f32")
        if(EXISTS "${FE_TEST_WEIGHTS}" AND EXISTS "${FE_TEST_AUDIO}")
            add_test(NAME fe_cross_tier_smoke
                     COMMAND fe_test_cross_tier
                             "${FE_TEST_WEIGHTS}" "${FE_TEST_AUDIO}" 32)
            add_test(NAME fe_run_file_smoke
                     COMMAND fe_run_file
                             "${FE_TEST_WEIGHTS}" "${FE_TEST_AUDIO}"
                             "${CMAKE_CURRENT_BINARY_DIR}/fe_smoke.wav" 32)
            set_tests_properties(fe_run_file_smoke PROPERTIES
                                 ENVIRONMENT "FE_RT=0")
        else()
            message(STATUS "  tests    : runners only (weights/testaudio .f32 missing)")
        endif()
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS fe
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/fe.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DOCDIR})
