"""
PHYSEC Flowgraphs Module

This module provides factory functions for creating GNU Radio flowgraphs
for the PHYSEC key generation system.
"""

# Individual component flowgraphs
from .physec_processor import create_physec_processor
from .sinusoidal_probe import create_sinusoidal_probe
from .iq_receiver import create_iq_receiver
from .parity_generator import create_parity_generator
from .reconciliator import create_reconciliator
from .privacy_amplifier import create_privacy_amplifier

# Comprehensive node flowgraphs
from .alice_node import create_alice_node_flowgraph
from .bob_node import create_bob_node_flowgraph

# Check if all flowgraphs are available by testing imports only
try:
    # Just test that we can import the modules, don't call functions
    import importlib
    import sys
    
    # Test that all required modules can be imported
    required_modules = [
        'flowgraphs.physec_processor',
        'flowgraphs.sinusoidal_probe',
        'flowgraphs.iq_receiver',
        'flowgraphs.parity_generator',
        'flowgraphs.reconciliator',
        'flowgraphs.privacy_amplifier',
        'flowgraphs.alice_node',
        'flowgraphs.bob_node'
    ]
    
    all_imported = True
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(f"⚠️  Module {module_name} import failed: {e}")
            all_imported = False
    
    if all_imported:
        FLOWGRAPHS_AVAILABLE = True
        print("✅ All PHYSEC flowgraph modules imported successfully")
    else:
        FLOWGRAPHS_AVAILABLE = False
        print("⚠️  Some PHYSEC flowgraph modules failed to import")
        
except Exception as e:
    FLOWGRAPHS_AVAILABLE = False
    print(f"⚠️  Error checking flowgraph availability: {e}")

# Export all factory functions
__all__ = [
    'create_physec_processor',
    'create_sinusoidal_probe', 
    'create_iq_receiver',
    'create_parity_generator',
    'create_reconciliator',
    'create_privacy_amplifier',
    'create_alice_node_flowgraph',
    'create_bob_node_flowgraph',
    'FLOWGRAPHS_AVAILABLE'
]

# Version information
__version__ = "1.0.0"
__author__ = "Jose Sanchez"
__description__ = "Modular GNU Radio flowgraphs for PHYSEC key generation"
