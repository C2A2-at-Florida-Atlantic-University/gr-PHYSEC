#!/usr/bin/env python3
"""
Enhanced PHYSEC Protocol Monitor
Monitors Alice and Bob nodes and displays their protocol status with IQ samples and spectrograms
"""

import socket
import json
import time
import argparse
import sys
import os
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dynamic_visualization import PhysecDynamicVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import visualization: {e}")
    VISUALIZATION_AVAILABLE = False

class EnhancedProtocolMonitor:
    def __init__(self, alice_ip, bob_ip, alice_port=9001, bob_port=9002):
        self.alice_ip = alice_ip
        self.bob_ip = bob_ip
        self.alice_port = alice_port
        self.bob_port = bob_port
        
        # Initialize visualization if available
        self.visualizer = None
        if VISUALIZATION_AVAILABLE:
            try:
                self.visualizer = PhysecDynamicVisualizer()
                self.visualizer.start_visualization()
                print("✅ Dynamic visualization started")
            except Exception as e:
                print(f"⚠️  Warning: Could not start visualization: {e}")
                self.visualizer = None
        
        # Node states
        self.alice_state = "unknown"
        self.bob_state = "unknown"
        self.alice_protocol_step = "Idle"
        self.bob_protocol_step = "Idle"
        self.current_run = 0
        
        # Data tracking
        self.alice_iq_available = False
        self.bob_iq_available = False
        self.alice_spectrogram_available = False
        self.bob_spectrogram_available = False
        
        # Statistics tracking
        self.alice_latest_bdr = None
        self.alice_latest_success = None
        self.alice_latest_timing_ms = None
        self.alice_total_runs = 0
        self.alice_successful_runs = 0
        
        self.bob_latest_bdr = None
        self.bob_latest_success = None
        self.bob_latest_timing_ms = None
        self.bob_total_runs = 0
        self.bob_successful_runs = 0
        
        # Quantized bits status tracking
        self.alice_has_quantized_bits = False
        self.bob_has_quantized_bits = False
        
        # Quantized bits data storage for BDR calculation
        self.alice_quantized_bits = None
        self.bob_quantized_bits = None
        
        # Data storage for visualization
        self.alice_iq_data = None
        self.alice_spectrogram_data = None
        self.bob_iq_data = None
        self.bob_spectrogram_data = None
        
        self.running = True
        
        print(f"🔧 Starting Enhanced PHYSEC Protocol Monitor...")
        print(f"📡 Monitoring Alice: {alice_ip}:{alice_port} (monitoring port)")
        print(f"📡 Monitoring Bob:   {bob_ip}:{bob_port} (monitoring port)")
        print(f"💡 Main protocol ports: Alice {alice_port-1000}, Bob {bob_port-1000}")
        
        if self.visualizer:
            print("🎨 Dynamic visualization enabled")
            print("   • Real-time protocol step updates")
            print("   • IQ samples and spectrogram visualization")
            print("   • Success rate and timing statistics")
        else:
            print("⚠️  Visualization disabled - text-only monitoring")
        
        print("⏳ Starting protocol monitoring...")

    def connect_to_node(self, ip, port, node_name):
        """Connect to get node status"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # 5 second timeout for connection
            sock.connect((ip, port))
            
            # Send status request
            request = {"type": "status_request"}
            sock.send(json.dumps(request).encode('utf-8') + b'\n')
            
            # Get response with timeout
            sock.settimeout(3)  # 3 second timeout for reading
            data = sock.recv(1024)
            sock.close()
            
            if data:
                try:
                    response = json.loads(data.decode('utf-8').strip())
                    return response
                except json.JSONDecodeError as e:
                    return {"state": "json_error", "error": f"Invalid JSON: {str(e)}"}
            else:
                return {"state": "no_response"}
                
        except socket.timeout:
            return {"state": "timeout", "error": "Connection or read timeout"}
        except Exception as e:
            return {"state": "connection_error", "error": str(e)}

    def request_data_from_node(self, ip, port, node_name, data_type):
        """Request specific data from a node"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # Longer timeout for data requests
            sock.connect((ip, port))
            
            # Send data request
            request = {"type": f"{data_type}_request"}
            request_str = json.dumps(request) + '\n'
            sock.send(request_str.encode('utf-8'))
            
            # Get response - handle large data properly
            response_data = b""
            sock.settimeout(2)  # Shorter timeout for reading
            
            while True:
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response_data += chunk
                    
                    # Check if we have a complete JSON response (ends with newline)
                    if b'\n' in response_data:
                        break
                        
                except socket.timeout:
                    # If we got some data and it looks complete, break
                    if response_data and b'\n' in response_data:
                        break
                    break
            
            sock.close()
            
            if response_data:
                # Find the complete JSON response (up to newline)
                if b'\n' in response_data:
                    response_data = response_data.split(b'\n')[0]
                
                try:
                    response = json.loads(response_data.decode('utf-8').strip())
                    return response
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error from {node_name}: {e}")
                    return {"error": f"json_decode_failed: {str(e)}"}
            else:
                return {"error": "no_data_response"}
                
        except Exception as e:
            print(f"❌ Error requesting {data_type} from {node_name}: {e}")
            return {"error": f"data_request_failed: {str(e)}"}

    def update_visualization(self):
        """Enhanced visualization update with data"""
        if self.visualizer:
            try:
                # Update protocol steps
                self.visualizer.update_step("Alice", self.alice_protocol_step)
                self.visualizer.update_step("Bob", self.bob_protocol_step)
                
                # Process events and update display
                self.visualizer.process_events()
                self.visualizer.update_display()
                
            except Exception as e:
                print(f"⚠️  Visualization update error: {e}")

    def update_visualization_with_data(self, node_name, iq_data=None, spectrogram_data=None):
        """Update visualization with actual data"""
        if self.visualizer:
            try:
                # Update IQ data if available
                if iq_data is not None:
                    self.visualizer.update_iq_data(node_name, iq_data)
                    # Store data locally
                    if node_name == "Alice":
                        self.alice_iq_data = iq_data
                    elif node_name == "Bob":
                        self.bob_iq_data = iq_data
                
                # Update spectrogram if available
                if spectrogram_data is not None:
                    self.visualizer.update_spectrogram(node_name, spectrogram_data)
                    # Store data locally
                    if node_name == "Alice":
                        self.alice_spectrogram_data = spectrogram_data
                    elif node_name == "Bob":
                        self.bob_spectrogram_data = spectrogram_data
                
                # Force visualization update
                self.visualizer.process_events()
                self.visualizer.update_display()
                
            except Exception as e:
                print(f"⚠️  Data visualization update error: {e}")
                
    def collect_available_data(self, node_name, ip, port):
        """Aggressively collect all available data from a node"""
        try:
            # Always try to get IQ samples if we don't have them
            if node_name == "Alice" and self.alice_iq_available and self.alice_iq_data is None:
                iq_response = self.request_data_from_node(ip, port, node_name, "iq_samples")
                if "iq_samples" in iq_response:
                    try:
                        iq_data = np.array(eval(iq_response["iq_samples"]), dtype=np.complex64)
                        self.update_visualization_with_data(node_name, iq_data=iq_data)
                        print(f"✅ {node_name} IQ samples collected: {len(iq_data)} samples")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} IQ data: {e}")
            
            # Always try to get spectrogram if we don't have it
            if node_name == "Alice" and self.alice_spectrogram_available and self.alice_spectrogram_data is None:
                spec_response = self.request_data_from_node(ip, port, node_name, "spectrogram")
                if "spectrogram_data" in spec_response:
                    try:
                        spec_data = np.array(eval(spec_response["spectrogram_data"]), dtype=np.float32)
                        self.update_visualization_with_data(node_name, spectrogram_data=spec_data)
                        print(f"✅ {node_name} spectrogram collected: shape {spec_data.shape}")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} spectrogram data: {e}")
            
            # Always try to get quantized bits if we don't have them
            if node_name == "Alice" and self.alice_has_quantized_bits and self.alice_quantized_bits is None:
                qb_response = self.request_data_from_node(ip, port, node_name, "quantized_bits")
                if "quantized_bits" in qb_response:
                    try:
                        qb_data = bytes(qb_response["quantized_bits"])
                        self.alice_quantized_bits = qb_data
                        print(f"✅ {node_name} quantized bits collected: {len(qb_data)} bytes")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} quantized bits: {e}")
            
            # Same for Bob
            if node_name == "Bob" and self.bob_iq_available and self.bob_iq_data is None:
                iq_response = self.request_data_from_node(ip, port, node_name, "iq_samples")
                if "iq_samples" in iq_response:
                    try:
                        iq_data = np.array(eval(iq_response["iq_samples"]), dtype=np.complex64)
                        self.update_visualization_with_data(node_name, iq_data=iq_data)
                        print(f"✅ {node_name} IQ samples collected: {len(iq_data)} samples")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} IQ data: {e}")
            
            if node_name == "Bob" and self.bob_spectrogram_available and self.bob_spectrogram_data is None:
                spec_response = self.request_data_from_node(ip, port, node_name, "spectrogram")
                if "spectrogram_data" in spec_response:
                    try:
                        spec_data = np.array(eval(spec_response["spectrogram_data"]), dtype=np.float32)
                        self.update_visualization_with_data(node_name, spectrogram_data=spec_data)
                        print(f"✅ {node_name} spectrogram collected: shape {spec_data.shape}")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} spectrogram data: {e}")
            
            if node_name == "Bob" and self.bob_has_quantized_bits and self.bob_quantized_bits is None:
                qb_response = self.request_data_from_node(ip, port, node_name, "quantized_bits")
                if "quantized_bits" in qb_response:
                    try:
                        qb_data = bytes(qb_response["quantized_bits"])
                        self.bob_quantized_bits = qb_data
                        print(f"✅ {node_name} quantized bits collected: {len(qb_data)} bytes")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} quantized bits: {e}")
                
        except Exception as e:
            print(f"❌ Error collecting data from {node_name}: {e}")
    
    def calculate_bdr_if_ready(self):
        """Calculate BDR when both nodes have quantized bits"""
        if (self.alice_quantized_bits is not None and 
            self.bob_quantized_bits is not None):
            
            try:
                # Convert bytes to numpy arrays
                alice_bits = np.frombuffer(self.alice_quantized_bits, dtype=np.uint8)
                bob_bits = np.frombuffer(self.bob_quantized_bits, dtype=np.uint8)
                
                # Calculate BDR
                min_len = min(len(alice_bits), len(bob_bits))
                bdr = np.mean(alice_bits[:min_len] != bob_bits[:min_len])
                
                print(f"🎯 BDR Calculated: {bdr:.4f} (Alice: {len(alice_bits)} bits, Bob: {len(bob_bits)} bits)")
                
                # Update visualization with calculated BDR
                if self.visualizer:
                    # Use a placeholder success value (will be updated when reconciliation completes)
                    success = True  # Placeholder - will be updated with real reconciliation result
                    timing_ms = None  # Placeholder - will be updated with real timing
                    
                    self.visualizer.add_run_statistics(bdr, success, timing_ms)
                    self.visualizer.process_events()
                    self.visualizer.update_display()
                
                # Clear the bits to avoid recalculating
                self.alice_quantized_bits = None
                self.bob_quantized_bits = None
                
            except Exception as e:
                print(f"❌ Error calculating BDR: {e}")
    
    def update_visualization_statistics(self):
        """Update visualization with collected statistics"""
        if self.visualizer:
            try:
                # Update with latest statistics from both nodes
                if self.alice_latest_bdr is not None and self.alice_latest_success is not None:
                    self.visualizer.add_run_statistics(
                        self.alice_latest_bdr, 
                        self.alice_latest_success, 
                        self.alice_latest_timing_ms
                    )
                
                # Force visualization update
                self.visualizer.process_events()
                self.visualizer.update_display()
                
            except Exception as e:
                print(f"⚠️  Statistics visualization update error: {e}")

    def show_status(self):
        """Display current status with data availability"""
        print("\n" + "="*70)
        print(f"🔐 Enhanced PHYSEC Protocol Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        print(f"\n📱 Node States:")
        print(f"   Alice ({self.alice_ip}:{self.alice_port}): {self.alice_state}")
        print(f"   Bob   ({self.bob_ip}:{self.bob_port}): {self.bob_state}")
        print(f"   💡 Monitoring via dedicated ports (main protocol: {self.alice_port-1000}, {self.bob_port-1000})")
        
        print(f"\n🚀 Protocol Execution:")
        print(f"   Current Run: {self.current_run}")
        print(f"   Alice Protocol Step: {self.alice_protocol_step}")
        print(f"   Bob Protocol Step: {self.bob_protocol_step}")
        
        print(f"\n📊 Data Availability:")
        print(f"   Alice IQ Samples: {'✅ Available' if self.alice_iq_available else '❌ Not available'}")
        print(f"   Alice Spectrogram: {'✅ Available' if self.alice_spectrogram_available else '❌ Not available'}")
        print(f"   Bob IQ Samples: {'✅ Available' if self.bob_iq_available else '❌ Not available'}")
        print(f"   Bob Spectrogram: {'✅ Available' if self.bob_spectrogram_available else '❌ Not available'}")
        
        print(f"\n📈 Statistics:")
        print(f"   Alice - BDR: {self.alice_latest_bdr:.4f}" if self.alice_latest_bdr is not None else "   Alice - BDR: N/A")
        print(f"   Alice - Success: {'✅' if self.alice_latest_success else '❌'}" if self.alice_latest_success is not None else "   Alice - Success: N/A")
        print(f"   Alice - Timing: {self.alice_latest_timing_ms:.0f}ms" if self.alice_latest_timing_ms is not None else "   Alice - Timing: N/A")
        print(f"   Alice - Runs: {self.alice_successful_runs}/{self.alice_total_runs}")
        
        print(f"   Bob - BDR: {self.bob_latest_bdr:.4f}" if self.bob_latest_bdr is not None else "   Bob - BDR: N/A")
        print(f"   Bob - Success: {'✅' if self.bob_latest_success else '❌'}" if self.bob_latest_success is not None else "   Bob - Success: N/A")
        print(f"   Bob - Timing: {self.bob_latest_timing_ms:.0f}ms" if self.bob_latest_timing_ms is not None else "   Bob - Timing: N/A")
        print(f"   Bob - Runs: {self.bob_successful_runs}/{self.bob_total_runs}")
        
        print(f"\n🔍 Quantized Bits Status:")
        print(f"   Alice: {'✅ Has bits' if self.alice_has_quantized_bits else '❌ No bits'}")
        print(f"   Bob:   {'✅ Has bits' if self.bob_has_quantized_bits else '❌ No bits'}")
        if self.alice_quantized_bits is not None:
            print(f"   📊 Alice bits loaded: {len(self.alice_quantized_bits)} bytes")
        if self.bob_quantized_bits is not None:
            print(f"   📊 Bob bits loaded: {len(self.bob_quantized_bits)} bytes")
        
        print(f"\n🎯 Protocol Progress:")
        if self.alice_protocol_step == "Probe TX" and self.bob_protocol_step == "Probe TX":
            print(f"   📡 Both nodes transmitting probes")
        elif self.alice_protocol_step == "Sample Collection" or self.bob_protocol_step == "Sample Collection":
            print(f"   📊 Sample collection in progress")
        elif self.alice_protocol_step == "PHYSEC Processing" or self.bob_protocol_step == "PHYSEC Processing":
            print(f"   🔬 PHYSEC processing in progress")
        elif self.alice_protocol_step == "Reconciliation" or self.bob_protocol_step == "Reconciliation":
            print(f"   🔑 Key reconciliation in progress")
        elif self.alice_protocol_step == "Complete" or self.bob_protocol_step == "Complete":
            print(f"   ✅ Protocol run completed")
        else:
            print(f"   ⏸️  Protocol idle or unknown state")
        
        print("="*70)

    def monitor_once(self):
        """Single monitoring cycle with data requests"""
        # Check Alice
        try:
            response = self.connect_to_node(self.alice_ip, self.alice_port, "Alice")
            old_state = self.alice_state
            old_step = self.alice_protocol_step
            old_iq = self.alice_iq_available
            old_spec = self.alice_spectrogram_available
            
            self.alice_state = response.get("state", "unknown")
            self.alice_protocol_step = response.get("protocol_step", "Idle")
            self.alice_iq_available = response.get("iq_samples_available", False)
            self.alice_spectrogram_available = response.get("spectrogram_available", False)
            
            # Collect statistics
            self.alice_latest_bdr = response.get("latest_bdr")
            self.alice_latest_success = response.get("latest_success")
            self.alice_latest_timing_ms = response.get("latest_timing_ms")
            self.alice_total_runs = response.get("total_runs", 0)
            self.alice_successful_runs = response.get("successful_runs", 0)
            
            # Collect quantized bits status
            self.alice_has_quantized_bits = response.get("quantized_bits_available", False)
            
            if "run_number" in response:
                self.current_run = response["run_number"]
            
            # Check for changes
            state_changed = self.alice_state != old_state
            step_changed = self.alice_protocol_step != old_step
            iq_changed = self.alice_iq_available != old_iq
            spec_changed = self.alice_spectrogram_available != old_spec
            
            # Track old statistics for change detection
            old_bdr = self.alice_latest_bdr
            old_success = self.alice_latest_success
            old_timing = self.alice_latest_timing_ms
            
            if state_changed or step_changed:
                print(f"🔄 Alice: {old_state}→{self.alice_state}, {old_step}→{self.alice_protocol_step}")
                self.update_visualization()
            
            # Request data if newly available
            if iq_changed and self.alice_iq_available:
                iq_response = self.request_data_from_node(self.alice_ip, self.alice_port, "Alice", "iq_samples")
                if "iq_samples" in iq_response:
                    try:
                        # Convert string representation back to numpy array
                        iq_data = np.array(eval(iq_response["iq_samples"]), dtype=np.complex64)
                        self.update_visualization_with_data("Alice", iq_data=iq_data)
                    except Exception as e:
                        print(f"❌ Error processing Alice IQ data: {e}")
                elif "error" in iq_response:
                    print(f"❌ Alice IQ data request failed: {iq_response['error']}")
            
            if spec_changed and self.alice_spectrogram_available:
                spec_response = self.request_data_from_node(self.alice_ip, self.alice_port, "Alice", "spectrogram")
                if "spectrogram_data" in spec_response:
                    try:
                        # Convert string representation back to numpy array
                        spec_data = np.array(eval(spec_response["spectrogram_data"]), dtype=np.float32)
                        self.update_visualization_with_data("Alice", spectrogram_data=spec_data)
                    except Exception as e:
                        print(f"❌ Error processing Alice spectrogram data: {e}")
                elif "error" in spec_response:
                    print(f"❌ Alice spectrogram data request failed: {spec_response['error']}")
            
            # Request quantized bits if newly available
            if self.alice_has_quantized_bits and self.alice_quantized_bits is None:
                qb_response = self.request_data_from_node(self.alice_ip, self.alice_port, "Alice", "quantized_bits")
                if "quantized_bits" in qb_response:
                    try:
                        # Convert string representation back to bytes
                        qb_data = bytes(qb_response["quantized_bits"])
                        self.alice_quantized_bits = qb_data
                        print(f"✅ Alice quantized bits received: {len(qb_data)} bytes")
                    except Exception as e:
                        print(f"❌ Error processing Alice quantized bits: {e}")
                elif "error" in qb_response:
                    print(f"❌ Alice quantized bits request failed: {qb_response['error']}")
            
            # Always try to collect data if available (more aggressive approach)
            self.collect_available_data("Alice", self.alice_ip, self.alice_port)
                
        except Exception as e:
            print(f"❌ Error monitoring Alice: {e}")
        
        # Check Bob
        try:
            response = self.connect_to_node(self.bob_ip, self.bob_port, "Bob")
            old_state = self.bob_state
            old_step = self.bob_protocol_step
            old_iq = self.bob_iq_available
            old_spec = self.bob_spectrogram_available
            
            self.bob_state = response.get("state", "unknown")
            self.bob_protocol_step = response.get("protocol_step", "Idle")
            self.bob_iq_available = response.get("iq_samples_available", False)
            self.bob_spectrogram_available = response.get("spectrogram_available", False)
            
            # Collect statistics
            self.bob_latest_bdr = response.get("latest_bdr")
            self.bob_latest_success = response.get("latest_success")
            self.bob_latest_timing_ms = response.get("latest_timing_ms")
            self.bob_total_runs = response.get("total_runs", 0)
            self.bob_successful_runs = response.get("successful_runs", 0)
            
            # Collect quantized bits status
            self.bob_has_quantized_bits = response.get("quantized_bits_available", False)
            
            if "run_number" in response:
                self.current_run = response["run_number"]
            
            # Check for changes
            state_changed = self.bob_state != old_state
            step_changed = self.bob_protocol_step != old_step
            iq_changed = self.bob_iq_available != old_iq
            spec_changed = self.bob_spectrogram_available != old_spec
            
            # Track old statistics for change detection
            old_bdr = self.bob_latest_bdr
            old_success = self.bob_latest_success
            old_timing = self.bob_latest_timing_ms
            
            if state_changed or step_changed:
                print(f"🔄 Bob: {old_state}→{self.bob_state}, {old_step}→{self.bob_protocol_step}")
                self.update_visualization()
            
            # Request data if newly available
            if iq_changed and self.bob_iq_available:
                iq_response = self.request_data_from_node(self.bob_ip, self.bob_port, "Bob", "iq_samples")
                if "iq_samples" in iq_response:
                    try:
                        # Convert string representation back to numpy array
                        iq_data = np.array(eval(iq_response["iq_samples"]), dtype=np.complex64)
                        self.update_visualization_with_data("Bob", iq_data=iq_data)
                    except Exception as e:
                        print(f"❌ Error processing Bob IQ data: {e}")
                elif "error" in iq_response:
                    print(f"❌ Bob IQ data request failed: {iq_response['error']}")
            
            if spec_changed and self.bob_spectrogram_available:
                spec_response = self.request_data_from_node(self.bob_ip, self.bob_port, "Bob", "spectrogram")
                if "spectrogram_data" in spec_response:
                    try:
                        # Convert string representation back to numpy array
                        spec_data = np.array(eval(spec_response["spectrogram_data"]), dtype=np.float32)
                        self.update_visualization_with_data("Bob", spectrogram_data=spec_data)
                    except Exception as e:
                        print(f"❌ Error processing Bob spectrogram data: {e}")
                elif "error" in spec_response:
                    print(f"❌ Bob spectrogram data request failed: {spec_response['error']}")
            
            # Request quantized bits if newly available
            if self.bob_has_quantized_bits and self.bob_quantized_bits is None:
                qb_response = self.request_data_from_node(self.bob_ip, self.bob_port, "Bob", "quantized_bits")
                if "quantized_bits" in qb_response:
                    try:
                        # Convert string representation back to bytes
                        qb_data = bytes(qb_response["quantized_bits"])
                        self.bob_quantized_bits = qb_data
                        print(f"✅ Bob quantized bits received: {len(qb_data)} bytes")
                    except Exception as e:
                        print(f"❌ Error processing Bob quantized bits: {e}")
                elif "error" in qb_response:
                    print(f"❌ Bob quantized bits request failed: {qb_response['error']}")
            
            # Always try to collect data if available (more aggressive approach)
            self.collect_available_data("Bob", self.bob_ip, self.bob_port)
                
        except Exception as e:
            print(f"❌ Error monitoring Bob: {e}")
        
        # Calculate BDR if both nodes have quantized bits
        self.calculate_bdr_if_ready()
        
        # Update visualization with any new statistics
        try:
            self.update_visualization_statistics()
        except Exception as e:
            print(f"⚠️  Statistics update error: {e}")

    def start_monitoring(self):
        """Start enhanced monitoring loop"""
        print(f"\n🔄 Enhanced monitoring started... (Press Ctrl+C to stop)")
        print(f"💡 Data visualization enabled - IQ samples and spectrograms will be displayed")
        
        try:
            while self.running:
                # Single monitoring cycle
                self.monitor_once()
                
                # Show status every 5 seconds
                time.sleep(5)
                self.show_status()
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping enhanced protocol monitor...")
        finally:
            if self.visualizer:
                try:
                    self.visualizer.stop_visualization()
                    print("🎨 Visualization stopped")
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description="Enhanced PHYSEC Protocol Monitor")
    parser.add_argument("--alice-ip", required=True, help="Alice's IP address")
    parser.add_argument("--bob-ip", required=True, help="Bob's IP address")
    parser.add_argument("--alice-port", type=int, default=9001, help="Alice's monitoring port (default: 9001)")
    parser.add_argument("--bob-port", type=int, default=9002, help="Bob's monitoring port (default: 9002)")
    
    args = parser.parse_args()
    
    monitor = EnhancedProtocolMonitor(args.alice_ip, args.bob_ip, args.alice_port, args.bob_port)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
