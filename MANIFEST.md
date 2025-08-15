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
- python/PHYSEC/spectrogram_block.py - Spectrogram generation block
- python/PHYSEC/feature_extraction_block.py - ONNX feature extraction block
- python/PHYSEC/feature_quantization_block.py - Binary quantization block
- python/PHYSEC/parity_generation_block.py - Reed-Solomon parity generation block
- python/PHYSEC/reconciliation_block.py - Key reconciliation block
- python/PHYSEC/privacy_amplification_block.py - Privacy amplification block
- python/PHYSEC/CMakeLists.txt - Python module build configuration
- python/PHYSEC/bindings/CMakeLists.txt - Bindings configuration

## GRC Files
- grc/CMakeLists.txt - GRC build configuration
- grc/PHYSEC_spectrogram_block.block.yml - Spectrogram block definition
- grc/PHYSEC_feature_extraction_block.block.yml - Feature extraction block definition
- grc/PHYSEC_feature_quantization_block.block.yml - Feature quantization block definition
- grc/PHYSEC_parity_generation_block.block.yml - Parity generation block definition
- grc/PHYSEC_reconciliation_block.block.yml - Reconciliation block definition
- grc/PHYSEC_privacy_amplification_block.block.yml - Privacy amplification block definition

## Documentation
- docs/CMakeLists.txt - Documentation build configuration
- docs/doxygen/CMakeLists.txt - Doxygen configuration

## Examples
- examples/decoupled_physic_example.grc - Example flowgraph using decoupled blocks
- examples/decoupled_physic_example.py - Example Python script using decoupled blocks

## Build Files
- cmake/cmake_uninstall.cmake.in - Uninstall script template
- cmake/Modules/gnuradio-PHYSECConfig.cmake - CMake configuration
- cmake/Modules/targetConfig.cmake.in - Target configuration template
