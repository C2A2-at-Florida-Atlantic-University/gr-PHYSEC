# gr-PHYSEC Manifest

This file lists all the files that are part of the gr-PHYSEC module.

## Core Files
- CMakeLists.txt - Main build configuration
- MANIFEST.md - This file
- README.md - Module documentation

## Include Files
- include/gnuradio/PHYSEC/PHYSEC.h - Main header file
- include/gnuradio/PHYSEC/api.h - API definitions

## Library Files
- lib/CMakeLists.txt - Library build configuration
- lib/PHYSEC_impl.cc - Library implementation

## Python Files
- python/PHYSEC/__init__.py - Python module initialization
- python/PHYSEC/fingerprint_block.py - Main fingerprinting block
- python/PHYSEC/CMakeLists.txt - Python module build configuration
- python/PHYSEC/bindings/CMakeLists.txt - Bindings configuration

## GRC Files
- grc/CMakeLists.txt - GRC build configuration
- grc/PHYSEC_fingerprint_block.block.yml - Block definition

## Documentation
- docs/CMakeLists.txt - Documentation build configuration
- docs/doxygen/CMakeLists.txt - Doxygen configuration

## Examples
- examples/fingerprint_example.grc - Example flowgraph
- examples/fingerprint_example.py - Example Python script

## Build Files
- cmake/cmake_uninstall.cmake.in - Uninstall script template
- cmake/Modules/gnuradio-PHYSECConfig.cmake - CMake configuration
- cmake/Modules/targetConfig.cmake.in - Target configuration template
