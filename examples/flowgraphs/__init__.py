"""
PHYSEC GNU Radio Flowgraphs Module

This module contains modular GNU Radio flowgraph implementations for the PHYSEC protocol.
Each flowgraph is self-contained and can be used independently or as part of the larger protocol.

Available Flowgraphs:
- PhysecProcessor: IQ signal processing through PHYSEC pipeline
- SinusoidalProbe: Sinusoidal signal generation and transmission
- IQReceiver: IQ sample collection and reception
- ParityGenerator: Reed-Solomon parity bit generation
- Reconciliator: Reed-Solomon reconciliation for key agreement
- PrivacyAmplifier: Final key generation through privacy amplification

Each flowgraph can be imported directly:
    from flowgraphs.physec_processor import create_physec_processor
    from flowgraphs.sinusoidal_probe import create_sinusoidal_probe
    # etc.

Or imported via factory functions:
    from flowgraphs import create_physec_processor, create_sinusoidal_probe
"""

# Import all factory functions for convenience
from .physec_processor import create_physec_processor, PhysecProcessor
from .sinusoidal_probe import create_sinusoidal_probe, SinusoidalProbe
from .iq_receiver import create_iq_receiver, IQReceiver
from .parity_generator import create_parity_generator, ParityGenerator
from .reconciliator import create_reconciliator, Reconciliator
from .privacy_amplifier import create_privacy_amplifier, PrivacyAmplifier

__all__ = [
    # Factory functions
    'create_physec_processor',
    'create_sinusoidal_probe', 
    'create_iq_receiver',
    'create_parity_generator',
    'create_reconciliator',
    'create_privacy_amplifier',
    
    # Classes
    'PhysecProcessor',
    'SinusoidalProbe',
    'IQReceiver', 
    'ParityGenerator',
    'Reconciliator',
    'PrivacyAmplifier'
]

# Version information
__version__ = "1.0.0"
__author__ = "Jose Sanchez"
__description__ = "Modular GNU Radio flowgraphs for PHYSEC quantum key generation"
