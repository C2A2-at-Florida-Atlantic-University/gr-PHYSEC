#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHYSEC Control Layer for Key Generation
Implements bidirectional TCP/IP communication between Alice and Bob nodes
for quantum key generation using GNU Radio and PlutoSDR.
"""

import socket
import threading
import time
import json
import numpy as np
from gnuradio import gr, blocks, analog, iio
from gnuradio import PHYSEC
import sys
import logging

try:
    from dynamic_visualization import get_visualizer
    DYNAMIC_VIZ_AVAILABLE = True
except ImportError:
    DYNAMIC_VIZ_AVAILABLE = False
    get_visualizer = None

from flowgraphs import (
    create_physec_processor,
    create_sinusoidal_probe, 
    create_iq_receiver,
    create_parity_generator,
    create_reconciliator,
    create_privacy_amplifier
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysecNode:
    """Base class for PHYSEC nodes (Alice/Bob)"""
    
    def __init__(self, node_name, listen_port, peer_host, peer_port, sdr_uri="ip:192.168.2.1"):
        self.node_name = node_name
        self.listen_port = listen_port
        self.peer_host = peer_host
        self.peer_port = peer_port
        self.sdr_uri = sdr_uri
        
        # SDR parameters
        self.sample_rate = 1000000
        self.center_freq = 2400000000
        self.gain = 30
        self.vector_size = 8192
        self.fft_window = 512
        self.key_length = 512
        self.k = 128
        self.n = 255
        
        # Communication
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.transmitting = False  # Flag for probe transmission control
        
        # Data storage
        self.iq_samples = None
        self.spectrogram_data = None
        self.features = None
        self.quantized_bits = None
        self.key = None
        
        # Multiple run tracking
        self.run_count = 0
        self.bit_disagreement_history = []
        self.correlation_history = []
        self.reconciliation_success_history = []
        
        # Active flowgraph tracking for cleanup
        self.active_probe_tx = None
        self.active_receiver = None
        
        # Dynamic visualization support
        self.visualizer = None
        if DYNAMIC_VIZ_AVAILABLE:
            self.visualizer = get_visualizer()
        
        # Pre-initialize flowgraphs for reuse
        self._initialize_flowgraphs()
        
        logger.info(f"Initialized {node_name} node - Listen: {listen_port}, Peer: {peer_host}:{peer_port}")

    def _initialize_flowgraphs(self):
        """Pre-initialize all flowgraphs for reuse"""
        logger.info(f"{self.node_name} initializing flowgraphs...")
        
        # Initialize signal source (probe transmitter)
        self.signal_source = self.create_signal_source()
        logger.debug(f"Signal source flowgraph initialized")
        
        # Signal receiver will be created on-demand to avoid state issues
        self.signal_receiver = None
        logger.debug(f"Signal receiver set to be created on-demand")
        
        # PHYSEC processor will be created on-demand since it needs real IQ samples
        self.physec_processor = None
        self.iq_samples = None
        logger.debug(f"PHYSEC processor set to be created on-demand")
        # Cryptographic flowgraphs will be created on-demand since they need real data
        self.parity_generator = None
        self.reconciliator = None
        self.privacy_amplifier = None
        logger.debug(f"Cryptographic flowgraphs set to be created on-demand")
        
        logger.info(f"{self.node_name} flowgraph initialization complete")

    def update_visualization_step(self, step):
        """Update the visualization with current protocol step"""
        if self.visualizer:
            self.visualizer.update_step(self.node_name, step)

    def update_visualization_iq(self, iq_data):
        """Update the visualization with IQ data"""
        if self.visualizer and iq_data is not None:
            self.visualizer.update_iq_data(self.node_name, iq_data)

    def update_visualization_spectrogram(self, spec_data):
        """Update the visualization with spectrogram data"""
        if self.visualizer and spec_data is not None:
            self.visualizer.update_spectrogram(self.node_name, spec_data)

    def start_server(self, ip="0.0.0.0"):
        """Start TCP server to listen for incoming connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ip, self.listen_port))
        self.server_socket.listen(1)
        logger.info(f"{self.node_name} server listening on port {self.listen_port}")
        
        self.running = True
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"{self.node_name} accepted connection from {addr}")
                threading.Thread(target=self.handle_client, args=(client_socket,)).start()
            except Exception as e:
                if self.running:
                    logger.error(f"Server error: {e}")

    def handle_client(self, client_socket):
        """Handle incoming client messages"""
        buffer = b""
        try:
            peer_info = client_socket.getpeername()
            logger.info(f"{self.node_name} handling messages from {peer_info}")
            
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    logger.info(f"{self.node_name} peer {peer_info} disconnected")
                    break
                
                buffer += data
                # Process complete messages (delimited by newlines)
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if line:
                        try:
                            message = json.loads(line.decode('utf-8'))
                            logger.info(f"{self.node_name} received: {message.get('type', 'unknown')} from {peer_info}")
                            self.process_message(message)
                        except json.JSONDecodeError as e:
                            logger.error(f"{self.node_name} failed to decode message from {peer_info}: {e}")
                    
        except Exception as e:
            logger.error(f"{self.node_name} client handler error: {e}")
        finally:
            try:
                client_socket.close()
                logger.info(f"{self.node_name} closed connection to {peer_info}")
            except:
                pass

    def send_message(self, message):
        """Send message to peer node"""
        try:
            if not self.client_socket:
                logger.info(f"{self.node_name} creating connection to {self.peer_host}:{self.peer_port}")
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.settimeout(10.0)  # Add timeout
                self.client_socket.connect((self.peer_host, self.peer_port))
                logger.info(f"{self.node_name} connected to peer successfully")
            
            data = json.dumps(message).encode('utf-8') + b'\n'  # Add newline delimiter
            self.client_socket.send(data)
            logger.info(f"{self.node_name} sent: {message['type']} to {self.peer_host}:{self.peer_port}")
            
        except Exception as e:
            logger.error(f"{self.node_name} failed to send message to {self.peer_host}:{self.peer_port}: {e}")
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            raise  # Re-raise to let caller handle the error

    def create_signal_source(self):
        """Create GNU Radio flowgraph for sinusoidal probe transmission"""
            # Use modular flowgraph
        return create_sinusoidal_probe(
            sample_rate=self.sample_rate,
            frequency=1000,  # 1 kHz probe tone
            amplitude=1.0,
            center_freq=self.center_freq,
            gain=self.gain,
            sdr_uri=self.sdr_uri
        )

    def create_signal_receiver(self):
        """Create GNU Radio flowgraph for IQ sample collection"""
        return create_iq_receiver(
            sample_rate=self.sample_rate,
            center_freq=self.center_freq,
            gain=self.gain,
            vector_size=self.vector_size,
            sdr_uri=self.sdr_uri
        )
        
    def transmit_probe(self):
        """Transmit sinusoidal probe signal with handshake control"""
        logger.info(f"{self.node_name} starting probe transmission...")
        
        try:
            # Use pre-initialized signal source
            self.signal_source.start_transmission()
            self.transmitting = True
            
            # Keep transmitting until told to stop
            logger.info(f"Sinusoidal probe transmission started (freq=1000Hz)")
            while self.transmitting:
                time.sleep(0.1)  # Small sleep to avoid busy waiting
            
        except Exception as e:
            logger.error(f"{self.node_name} probe transmission failed: {e}")
            
        finally:
            # Always cleanup, even if there was an error
            self.transmitting = False
            if self.signal_source:
                try:
                    self.signal_source.stop_transmission()
                except Exception as e:
                    logger.warning(f"{self.node_name} error during cleanup: {e}")
        
        logger.info(f"{self.node_name} finished probe transmission")
    
    def stop_transmission(self):
        """Stop ongoing probe transmission"""
        self.transmitting = False

    def collect_samples(self):
        """Collect IQ samples from PlutoSDR"""
        logger.info(f"{self.node_name} collecting IQ samples...")
        self.update_visualization_step("Sample Collection")
        
        try:
            # Create fresh receiver for each sample collection to avoid state issues
            if self.signal_receiver is None:
                self.signal_receiver = self.create_signal_receiver()
            
            
            self.signal_receiver.start_reception()
            
            # Wait for samples to be collected
            time.sleep(3.0)  # Longer timeout for reliable collection
            
            # Get collected data
            if hasattr(self.signal_receiver, 'get_samples'):
                # Use modular flowgraph method
                self.iq_samples = self.signal_receiver.get_samples()
            else:
                # Fallback to direct vector sink access
                data = self.signal_receiver.vector_sink.data()
                if data:
                    self.iq_samples = np.array(data, dtype=np.complex64)
                else:
                    self.iq_samples = None
            
            if self.iq_samples is not None and len(self.iq_samples) > 0:
                logger.info(f"{self.node_name} collected {len(self.iq_samples)} samples")
                # Update visualization with IQ data
                self.update_visualization_iq(self.iq_samples)
            else:
                logger.error(f"{self.node_name} failed to collect samples")
                
        finally:
            # Always cleanup, even if there was an error
            if self.signal_receiver:
                try:
                    self.signal_receiver.stop_reception()
                except Exception as e:
                    logger.warning(f"{self.node_name} error stopping reception: {e}")

    def process_physec_pipeline(self):
        """Process samples through PHYSEC pipeline"""
        if self.iq_samples is None:
            logger.error(f"{self.node_name} no samples to process")
            return
            
        logger.info(f"{self.node_name} processing samples through PHYSEC pipeline...")
        self.update_visualization_step("PHYSEC Processing")
        
        # Create or recreate PHYSEC processor with current samples
        if self.physec_processor is not None:
            try:
                self.physec_processor.cleanup()
                del self.physec_processor
            except:
                pass
        
        # Create new processor with real IQ samples
        self.physec_processor = create_physec_processor(
            samples=self.iq_samples,
            fft_window=self.fft_window,
            vector_size=self.vector_size
        )
        
        try:
            self.physec_processor.start()
            self.physec_processor.wait()
            
            # Read quantized bits
            # Use modular flowgraph methods
            self.quantized_bits = self.physec_processor.get_quantized_bits()
            self.spectrogram_data = self.physec_processor.get_spectrogram_data()
            
            if self.quantized_bits:
                logger.info(f"{self.node_name} extracted {len(self.quantized_bits)} quantized bits")
            if self.spectrogram_data is not None:
                logger.info(f"{self.node_name} processed spectrogram data with shape {self.spectrogram_data.shape}")
                # Update visualization with spectrogram
                self.update_visualization_spectrogram(self.spectrogram_data)                
        finally:
            # Ensure proper cleanup of GNU Radio flowgraph
            try:
                self.physec_processor.stop()
                self.physec_processor.wait()
                self.physec_processor.cleanup()  # Call modular flowgraph cleanup
            except:
                pass

    def generate_parity_bits(self):
        """Generate parity bits for reconciliation"""
        if self.quantized_bits is None:
            logger.error(f"{self.node_name} no quantized bits for parity generation")
            return None
        
        logger.info(f"{self.node_name} generating parity bits...")
        
        # Recreate parity generator with real quantized bits
        if self.parity_generator is None:
            self.parity_generator = create_parity_generator(self.quantized_bits, n=self.n, k=self.k)
        
        try:
            self.parity_generator.start()
            self.parity_generator.wait()
            
            # Get parity bits using the modular method
            parity_bits = self.parity_generator.get_parity_bits()
            
            if parity_bits:
                logger.info(f"{self.node_name} generated {len(parity_bits)} parity bits")
                return parity_bits
            else:
                logger.error(f"{self.node_name} failed to generate parity bits")
                return None
                
        finally:
            try:
                self.parity_generator.stop()
                self.parity_generator.wait()
                self.parity_generator.cleanup()
            except:
                pass

    def perform_reconciliation(self, parity_bits):
        """Perform reconciliation using received parity bits"""
        if self.quantized_bits is None or parity_bits is None:
            logger.error(f"{self.node_name} missing data for reconciliation")
            return False
        
        logger.info(f"{self.node_name} performing reconciliation...")
        
        # Recreate reconciliator with real data
        if self.reconciliator is None:
            self.reconciliator = create_reconciliator(self.quantized_bits, parity_bits)
            
        try:
            self.reconciliator.start()
            self.reconciliator.wait()
            
            # Get reconciliation results using modular methods
            reconciled_key = self.reconciliator.get_reconciled_key()
            success = self.reconciliator.get_success_flag()
            
            if success and reconciled_key is not None and len(reconciled_key) > 0:
                logger.info(f"{self.node_name} reconciliation successful")
                return True
            else:
                logger.warning(f"{self.node_name} reconciliation failed (success: {success}, key length: {len(reconciled_key) if reconciled_key else 0})")
                return False
                
        finally:
            try:
                self.reconciliator.stop()
                self.reconciliator.wait()
                self.reconciliator.cleanup()
            except:
                pass
    
    def perform_privacy_amplification(self):
        """Perform privacy amplification to generate final key"""
        logger.info(f"{self.node_name} performing privacy amplification...")
        
        # Get reconciled key data
        reconciled_key = None
        
        # Check if we have a shared reconciled key (Alice gets this from Bob)
        if hasattr(self, 'shared_reconciled_key') and self.shared_reconciled_key is not None:
            reconciled_key = self.shared_reconciled_key
            logger.info(f"{self.node_name} using shared reconciled key ({len(reconciled_key)} bytes)")
        # Otherwise try to get from own reconciliation flowgraph (Bob case)
        elif self.reconciliator is not None:
            try:
                reconciled_key = self.reconciliator.get_reconciled_key()
                if reconciled_key is None or len(reconciled_key) == 0:
                    logger.error(f"{self.node_name} no reconciled key data available")
                    return False
                logger.info(f"{self.node_name} got {len(reconciled_key)} byte reconciled key")
            except Exception as e:
                logger.error(f"Failed to get reconciled key: {e}")
                return False
        else:
            logger.error(f"{self.node_name} no reconciled key available")
            return False

        # Create or update privacy amplifier with reconciled key
        if self.privacy_amplifier is None:
            self.privacy_amplifier = create_privacy_amplifier(reconciled_key)
        else:
            self.privacy_amplifier.update_key_data(reconciled_key)
            
        try:
            self.privacy_amplifier.start()
            self.privacy_amplifier.wait()
            
            # Get final key using modular method
            self.key = self.privacy_amplifier.get_final_key()
            
            if self.key:
                logger.info(f"{self.node_name} generated {len(self.key)} byte key")
                return True
            else:
                logger.error(f"{self.node_name} failed to generate final key")
                return False
                
        finally:
            try:
                self.privacy_amplifier.stop()
                self.privacy_amplifier.wait()
                self.privacy_amplifier.cleanup()
            except:
                pass

    def process_message(self, message):
        """Process incoming messages - to be overridden by subclasses"""
        pass

    def reset_for_new_run(self):
        """Reset state for a new protocol run"""
        # Cleanup any active flowgraphs first
        self.cleanup_active_flowgraphs()
        
        # Reset data
        self.iq_samples = None
        self.spectrogram_data = None
        self.features = None
        self.quantized_bits = None
        self.key = None
        self.shared_reconciled_key = None
        self.state = "idle"
        self.transmitting = False  # Reset transmission flag
        self.run_count += 1
        
        # Give hardware and GNU Radio blocks time to fully release resources
        # Moderate delay for PlutoSDR to fully disconnect
        time.sleep(3.0)
        
        logger.info(f"{self.node_name} reset for run #{self.run_count}")
    
    def cleanup_active_flowgraphs(self):
        """Clean up any active GNU Radio flowgraphs"""
        if self.active_probe_tx:
            try:
                if hasattr(self.active_probe_tx, 'stop_transmission'):
                    self.active_probe_tx.stop_transmission()
                elif hasattr(self.active_probe_tx, 'is_running') and self.active_probe_tx.is_running:
                    self.active_probe_tx.stop()
                    self.active_probe_tx.wait()
            except Exception as e:
                logger.warning(f"Error stopping probe TX: {e}")
            self.active_probe_tx = None
        
        if self.active_receiver:
            try:
                if hasattr(self.active_receiver, 'stop_reception'):
                    self.active_receiver.stop_reception()
                elif hasattr(self.active_receiver, 'is_running') and self.active_receiver.is_running:
                    self.active_receiver.stop()
                    self.active_receiver.wait()
            except Exception as e:
                logger.warning(f"Error stopping receiver: {e}")
            self.active_receiver = None
            
                # Clean up temporary files from previous runs
        import os
        temp_files = ['/tmp/quantized_features.txt', '/tmp/parity_bits.txt', 
                      '/tmp/reconciled_bits.txt', '/tmp/reconciliation_success.txt', '/tmp/final_key.txt']
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Error removing {temp_file}: {e}")
        
        # Cleanup pre-initialized flowgraphs
        self._cleanup_flowgraphs()
        
    def _cleanup_flowgraphs(self):
        """Clean up all pre-initialized flowgraphs"""
        flowgraphs = [
            ('signal_source', self.signal_source),
            ('signal_receiver', self.signal_receiver),
            ('physec_processor', self.physec_processor),
            ('parity_generator', self.parity_generator),
            ('reconciliator', self.reconciliator),
            ('privacy_amplifier', self.privacy_amplifier)
        ]
        
        for name, flowgraph in flowgraphs:
            if flowgraph is not None:
                try:
                    if hasattr(flowgraph, 'stop'):
                        flowgraph.stop()
                    if hasattr(flowgraph, 'wait'):
                        flowgraph.wait()
                    if hasattr(flowgraph, 'cleanup'):
                        flowgraph.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {e}")

    def update_statistics(self, other_node):
        """Update running statistics from this protocol run"""
        if (self.quantized_bits is not None and other_node.quantized_bits is not None):
            # Calculate bit disagreement for this run
            alice_bits = np.frombuffer(self.quantized_bits, dtype=np.uint8)
            bob_bits = np.frombuffer(other_node.quantized_bits, dtype=np.uint8)
            
            min_len = min(len(alice_bits), len(bob_bits))
            disagreement_ratio = np.mean(alice_bits[:min_len] != bob_bits[:min_len])
            
            # Calculate IQ correlation
            if self.iq_samples is not None and other_node.iq_samples is not None:
                alice_mag = np.abs(self.iq_samples)
                bob_mag = np.abs(other_node.iq_samples)
                min_samples = min(len(alice_mag), len(bob_mag))
                correlation = np.corrcoef(alice_mag[:min_samples], bob_mag[:min_samples])[0, 1]
            else:
                correlation = 0.0
            
            # Store in both nodes
            self.bit_disagreement_history.append(disagreement_ratio)
            self.correlation_history.append(correlation)
            other_node.bit_disagreement_history.append(disagreement_ratio)
            other_node.correlation_history.append(correlation)
            
            # Track reconciliation success
            success = self.state == "key_ready" and other_node.state == "key_ready"
            self.reconciliation_success_history.append(success)
            other_node.reconciliation_success_history.append(success)
            
            logger.info(f"Run #{self.run_count} - BER: {disagreement_ratio:.4f}, Correlation: {correlation:.4f}, Success: {success}")


    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        # Cleanup GNU Radio flowgraphs first
        self.cleanup_active_flowgraphs()
        
        # Cleanup network resources
        if self.server_socket:
            self.server_socket.close()
        if self.client_socket:
            self.client_socket.close()


class Alice(PhysecNode):
    """Alice node implementation"""
    
    def __init__(self, listen_port=8001, peer_host='localhost', peer_port=8002):
        super().__init__("Alice", listen_port, peer_host, peer_port)
        self.state = "idle"

    def start_key_generation(self):
        """Initiate key generation process"""
        logger.info("Alice initiating key generation...")
        self.state = "requesting"
        self.update_visualization_step("Key Request")
        
        try:
            # Step 1: Send key generation request
            message = {
                "type": "key_generation_request",
                "timestamp": time.time()
            }
            self.send_message(message)
            logger.info("Alice successfully sent key generation request")
        except Exception as e:
            logger.error(f"Alice failed to send key generation request: {e}")
            self.state = "error"
            raise

    def process_message(self, message):
        """Process messages specific to Alice"""
        msg_type = message.get("type")
        
        if msg_type == "probe_transmission_started" and self.state == "requesting":
            # Step 3: Bob started transmitting, acknowledge and collect samples
            logger.info("Alice received probe transmission notification")
            
            # Send acknowledgment that Alice will start collecting
            response = {
                "type": "collection_started",
                "timestamp": time.time()
            }
            self.send_message(response)
            
            # Small delay to ensure acknowledgment is sent
            time.sleep(0.2)
            
            # Collect samples from Bob's transmission
            self.collect_samples()
            
            # Tell Bob to stop transmitting
            response = {
                "type": "stop_transmission",
                "timestamp": time.time()
            }
            self.send_message(response)
            
            # Start own transmission
            response = {
                "type": "probe_transmission_started",
                "timestamp": time.time()
            }
            self.send_message(response)
            self.state = "transmitting"
            self.update_visualization_step("Probe TX")
            
            # Start own transmission (after notification to avoid blocking)
            threading.Thread(target=self.transmit_probe, name="AliceProbeTransmission").start()
            
        elif msg_type == "stop_transmission" and self.state == "transmitting":
            # Bob finished collecting samples, stop transmission
            logger.info("Alice received stop transmission request")
            self.stop_transmission()
            
        elif msg_type == "samples_collected" and self.state == "transmitting":
            # Step 4: Bob collected samples, start processing
            logger.info("Alice received samples collected notification")
            self.state = "processing"
            
            # Process through PHYSEC pipeline
            self.process_physec_pipeline()
            
            # Step 6: Generate and send parity bits
            self.update_visualization_step("Parity Generation")
            parity_bits = self.generate_parity_bits()
            if parity_bits:
                message = {
                    "type": "parity_bits",
                    "data": list(parity_bits),  # Convert bytes to list for JSON
                    "timestamp": time.time()
                }
                self.send_message(message)
                self.state = "sent_parity"
                
        elif msg_type == "reconciliation_result":
            # Step 8: Handle reconciliation result
            success = message.get("success", False)
            if success:
                logger.info("Alice received successful reconciliation notification")
                
                # Get shared reconciled key from Bob
                reconciled_key_hex = message.get("reconciled_key")
                if reconciled_key_hex:
                    try:
                        self.shared_reconciled_key = bytes.fromhex(reconciled_key_hex)
                        logger.info(f"Alice received {len(self.shared_reconciled_key)} byte reconciled key from Bob")
                    except Exception as e:
                        logger.error(f"Alice failed to decode reconciled key: {e}")
                        self.shared_reconciled_key = None
                else:
                    logger.warning("Alice did not receive reconciled key from Bob")
                    self.shared_reconciled_key = None
                
                # Perform privacy amplification
                if self.perform_privacy_amplification():
                    self.state = "key_ready"
                    logger.info("Alice key generation completed successfully!")
                    
                    # Step 9: Send encrypted test message
                    self.send_encrypted_message("Hello from Alice!")
                    
                    # Signal run completion for statistics tracking
                    message = {
                        "type": "run_complete",
                        "run_number": self.run_count + 1,
                        "timestamp": time.time()
                    }
                    self.send_message(message)
            else:
                logger.warning("Alice received failed reconciliation - marking run as failed")
                self.state = "reconciliation_failed"
                
        elif msg_type == "encrypted_message":
            # Step 9: Receive and decrypt message
            encrypted_data = message.get("data")
            logger.info(f"Alice received encrypted message: {encrypted_data}")
            # In a real implementation, decrypt using self.key
            
        elif msg_type == "run_complete":
            # Handle run completion notification
            run_number = message.get("run_number", 0)
            logger.info(f"Alice received run completion notification for run #{run_number}")

    def send_encrypted_message(self, plaintext):
        """Send encrypted message to Bob"""
        # In a real implementation, encrypt using self.key
        encrypted_data = f"ENCRYPTED({plaintext})"
        
        message = {
            "type": "encrypted_message",
            "data": encrypted_data,
            "timestamp": time.time()
        }
        self.send_message(message)
        logger.info(f"Alice sent encrypted message: {plaintext}")


class Bob(PhysecNode):
    """Bob node implementation"""
    
    def __init__(self, listen_port=8002, peer_host='localhost', peer_port=8001):
        super().__init__("Bob", listen_port, peer_host, peer_port)
        self.state = "idle"

    def process_message(self, message):
        """Process messages specific to Bob"""
        msg_type = message.get("type")
        
        if msg_type == "key_generation_request" and self.state == "idle":
            # Step 2: Accept request and start transmitting probe
            logger.info("Bob received key generation request")
            self.state = "accepted"
            
            # Notify Alice that transmission will start
            response = {
                "type": "probe_transmission_started", 
                "timestamp": time.time()
            }
            self.send_message(response)
            self.state = "transmitting"
            
        elif msg_type == "collection_started" and self.state == "transmitting":
            # Alice acknowledged, start actual transmission
            logger.info("Bob starting probe transmission...")
            threading.Thread(target=self.transmit_probe).start()
            
        elif msg_type == "stop_transmission" and self.state == "transmitting":
            # Alice finished collecting, stop transmission
            logger.info("Bob received stop transmission request")
            self.stop_transmission()
            
        elif msg_type == "probe_transmission_started" and self.state == "transmitting":
            # Step 4: Alice started transmitting, acknowledge and collect samples
            logger.info("Bob received Alice's transmission notification")
            
            # Send acknowledgment that Bob will start collecting
            response = {
                "type": "collection_started",
                "timestamp": time.time()
            }
            self.send_message(response)
            
            # Small delay to ensure acknowledgment is sent
            time.sleep(0.2)
            
            # Collect samples from Alice's transmission
            self.collect_samples()
            
            # Tell Alice to stop transmitting
            response = {
                "type": "stop_transmission",
                "timestamp": time.time()
            }
            self.send_message(response)
            
            # Notify Alice that samples are collected
            response = {
                "type": "samples_collected",
                "timestamp": time.time()
            }
            self.send_message(response)
            self.state = "processing"
            
            # Process through PHYSEC pipeline
            self.process_physec_pipeline()
            
        elif msg_type == "parity_bits" and self.state == "processing":
            # Step 7: Receive parity bits and perform reconciliation
            logger.info("Bob received parity bits")
            parity_data = bytes(message.get("data", []))
            
            # Perform reconciliation
            success = self.perform_reconciliation(parity_data)
            
            # Get reconciled key for sharing with Alice
            reconciled_key_data = None
            if success and self.reconciliator is not None:
                try:
                    reconciled_key_data = self.reconciliator.get_reconciled_key()
                    if reconciled_key_data:
                        logger.info(f"Bob sharing {len(reconciled_key_data)} byte reconciled key with Alice")
                except Exception as e:
                    logger.warning(f"Bob failed to get reconciled key for sharing: {e}")
            
            # Notify Alice of reconciliation result and share reconciled key
            response = {
                "type": "reconciliation_result",
                "success": success,
                "reconciled_key": reconciled_key_data.hex() if reconciled_key_data else None,
                "timestamp": time.time()
            }
            self.send_message(response)
            
            if success:
                # Step 8: Perform privacy amplification
                if self.perform_privacy_amplification():
                    self.state = "key_ready"
                    logger.info("Bob key generation completed successfully!")
                    
                    # Step 9: Send encrypted test message
                    self.send_encrypted_message("Hello from Bob!")
            else:
                self.state = "idle"
                
        elif msg_type == "encrypted_message":
            # Step 9: Receive and decrypt message
            encrypted_data = message.get("data")
            logger.info(f"Bob received encrypted message: {encrypted_data}")
            # In a real implementation, decrypt using self.key
            
        elif msg_type == "run_complete":
            # Handle run completion - Bob responds to continue or stop
            run_number = message.get("run_number", 0)
            logger.info(f"Bob received run completion notification for run #{run_number}")
            
            # Send acknowledgment back to Alice
            response = {
                "type": "run_ack",
                "run_number": run_number,
                "timestamp": time.time()
            }
            self.send_message(response)

    def send_encrypted_message(self, plaintext):
        """Send encrypted message to Alice"""
        # In a real implementation, encrypt using self.key
        encrypted_data = f"ENCRYPTED({plaintext})"
        
        message = {
            "type": "encrypted_message",
            "data": encrypted_data,
            "timestamp": time.time()
        }
        self.send_message(message)
        logger.info(f"Bob sent encrypted message: {plaintext}")


def main():
    """Main function to run the control layer demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PHYSEC Control Layer Demo')
    parser.add_argument('--node', choices=['alice', 'bob'], required=True,
                       help='Node type to run')
    parser.add_argument('--peer-host', default='localhost',
                       help='Peer node hostname')
    
    args = parser.parse_args()
    
    try:
        if args.node == 'alice':
            node = Alice(peer_host=args.peer_host)
            
            # Start server in background
            server_thread = threading.Thread(target=node.start_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait a bit for server to start
            time.sleep(1)
            
            # Start key generation process
            node.start_key_generation()
            
            # Keep Alice alive to receive responses
            logger.info("Alice waiting for protocol completion...")
            try:
                while node.running and node.state not in ["key_ready", "error", "reconciliation_failed"]:
                    time.sleep(0.5)
                
                if node.state == "key_ready":
                    logger.info("✅ Alice key generation completed successfully!")
                    if hasattr(node, 'key') and node.key:
                        logger.info(f"✅ Alice generated {len(node.key)} byte key")
                elif node.state == "error":
                    logger.error("❌ Alice key generation failed with error")
                elif node.state == "reconciliation_failed":
                    logger.error("❌ Alice key generation failed during reconciliation")
                else:
                    logger.warning("⚠️  Alice key generation incomplete")
                    
            except KeyboardInterrupt:
                logger.info("Alice interrupted by user")
                node.running = False
            
        elif args.node == 'bob':
            node = Bob(peer_host=args.peer_host)
            
            # Start server and wait for connections
            node.start_server()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if 'node' in locals():
            node.cleanup()
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
