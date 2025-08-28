#!/usr/bin/env python3
"""
Simple Push-Only PHYSEC Protocol Monitor
Receives pushed data from nodes and visualizes it in real-time
"""

import socket
import json
import time
import argparse
import sys
import os
import numpy as np
import threading

try:
    import zmq
    import pmt
    ZMQ_AVAILABLE = True
except Exception:
    ZMQ_AVAILABLE = False

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dynamic_visualization import PhysecDynamicVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import visualization: {e}")
    VISUALIZATION_AVAILABLE = False

class SimplePushMonitor:
    def __init__(self, alice_ip, bob_ip):
        self.alice_ip = alice_ip
        self.bob_ip = bob_ip
        
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
        
        # Data storage for visualization
        self.alice_iq_data = None
        self.alice_spectrogram_data = None
        self.alice_quantized_bits = None
        self.bob_iq_data = None
        self.bob_spectrogram_data = None
        self.bob_quantized_bits = None
        
        # Protocol state tracking (from pushed data)
        self.alice_protocol_step = "Idle"
        self.bob_protocol_step = "Idle"
        self.current_run = 0
        self.timing = {
            'start_ts': None,
            'alice_done_ts': None,
            'bob_done_ts': None,
        }
        self.pending_bdr = None  # Hold BDR until timing is computed to avoid duplicate points
        
        # Statistics storage
        self.alice_stats = {}
        self.bob_stats = {}
        
        # Data collection server for receiving pushed data
        self.data_server_socket = None
        self.data_server_running = True
        self.message_count = 0  # Track total messages received
        
        
        # Trigger publisher (ZeroMQ PUB) to start runs
        self.trigger_pub = None
        self.trigger_pub_addr = os.getenv('TRIGGER_BIND_ADDR', 'tcp://192.168.0.142:9103')
        self.trigger_ready = False
        self.auto_trigger = False
        
        print(f"🔧 Starting Simple Push-Only PHYSEC Monitor...")
        print(f"📡 Will receive data from Alice: {alice_ip}")
        print(f"📡 Will receive data from Bob:   {bob_ip}")
        print(f"📡 Data collection server: Port 9999")
        
        if self.visualizer:
            print("🎨 Dynamic visualization enabled")
            print("   • Real-time data visualization")
            print("   • IQ samples and spectrograms")
            print("   • Success rate and timing statistics")
            print(f"   • Visualizer object: {self.visualizer}")
            print(f"   • Visualizer running: {self.visualizer.running if self.visualizer else 'N/A'}")
        else:
            print("⚠️  Visualization disabled - text-only monitoring")
        
        print("⏳ Starting data collection server...")
        
    def _normalize_protocol_step(self, node_name, step_value):
        """Normalize incoming protocol step strings across nodes.

        - Strips node-specific prefixes like 'alice_' and 'bob_'
        - Maps common aliases to canonical labels
        - Ignores PMT boolean prints like '#t' and '#f'
        """
        try:
            raw = str(step_value) if step_value is not None else ""
            if raw in ("#t", "#f", "", "None"):
                return None

            # Strip node prefixes
            if raw.startswith("alice_"):
                raw = raw[len("alice_"):]
            elif raw.startswith("bob_"):
                raw = raw[len("bob_"):]

            # Map raw tokens to the visualizer's canonical step labels
            # Visualizer expects: ["Idle","Key Request","Probe TX","Sample Collection","PHYSEC Processing","Parity Generation","Reconciliation","Privacy Amplification","Key Exchange","Complete"]
            step_map = {
                "probe_req": "Key Request",
                "accept": "Key Request",
                "start": "Key Request",  # RX gate start -> sampling begins
                "collecting": "Sample Collection",
                "rx_collected": "Sample Collection",
                "collect_done": "Sample Collection",
                "tx_start": "Probe TX",
                "tx_stop": "Probe TX",
                "processing": "PHYSEC Processing",
                "parity": "Parity Generation",
                "parity_received": "Reconciliation",
                "parity_recv": "Reconciliation",
                "reconcile_ok": "Complete",
                "reconcile_fail": "Complete",
                "exchange": "Key Exchange",
                "done": "Complete",
            }
            return step_map.get(raw, raw)
        except Exception:
            return None

    def _start_trigger_publisher(self):
        """Initialize ZMQ PUB to send 'start' triggers to nodes."""
        if not ZMQ_AVAILABLE:
            print("⚠️  ZeroMQ/PMT not available - will not auto-trigger runs")
            return
        try:
            ctx = zmq.Context.instance()
            pub = ctx.socket(zmq.PUB)
            bind_addr = self.trigger_pub_addr
            pub.bind(bind_addr)
            self.trigger_pub = pub
            self.trigger_ready = True
            print(f"✅ Trigger publisher bound at {bind_addr}")
        except Exception as e:
            print(f"❌ Failed to start trigger publisher: {e}")
            self.trigger_pub = None
            self.trigger_ready = False

    def _send_trigger_start(self, delay_s: float = 0.5):
        """Send a single 'start' trigger after optional delay."""
        if not (ZMQ_AVAILABLE and self.trigger_pub and self.trigger_ready):
            return
        def _send():
            try:
                if delay_s > 0:
                    time.sleep(delay_s)
                msg = pmt.serialize_str(pmt.intern('start'))
                self.trigger_pub.send(msg)
                print("📣 Sent trigger: start")
            except Exception as e:
                print(f"⚠️  Failed to send trigger: {e}")
        threading.Thread(target=_send, daemon=True).start()

    def start_data_collection_server(self):
        """Start server to receive pushed data from nodes"""
        try:
            self.data_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_server_socket.bind(('192.168.0.142', 9999))  # Use port 9999 for data collection
            self.data_server_socket.listen(5)
            
            print(f"✅ Data collection server listening on port 9999")
            
            while self.data_server_running:
                try:
                    client_socket, addr = self.data_server_socket.accept()
                    print(f"📡 Data connection from {addr}")
                    t = threading.Thread(target=self.handle_data_connection, args=(client_socket, addr), daemon=True)
                    t.start()
                except Exception as e:
                    if self.data_server_running:
                        print(f"❌ Data server error: {e}")
                        
        except Exception as e:
            print(f"❌ Failed to start data server: {e}")

    def handle_data_connection(self, client_socket, addr):
        """Handle data connection from a node"""
        try:
            print(f"🔗 Processing data connection from {addr}")
            buffer = b""  # Buffer for incomplete messages
            last_activity = time.time()
            
            while self.data_server_running:
                # Set timeout for receiving data
                client_socket.settimeout(5.0)  # 5 second timeout
                
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    
                    # Update activity timestamp
                    last_activity = time.time()
                    
                    # Add new data to buffer
                    buffer += data
                    
                    # Process complete messages from buffer
                    while b'\n' in buffer:
                        # Split on newline to get complete messages
                        message_data, buffer = buffer.split(b'\n', 1)
                        
                        if message_data.strip():
                            try:
                                message = json.loads(message_data.decode('utf-8').strip())
                                self.process_pushed_data(message, addr)
                            except json.JSONDecodeError as e:
                                print(f"⚠️  JSON decode error from {addr}: {e}")
                                print(f"   Message data: {message_data[:200]}...")
                                # Try to find where the JSON might be valid
                                if b'"type"' in message_data and b'"data_type"' in message_data:
                                    print(f"   🔍 Message appears to be truncated, waiting for more data...")
                                continue
                    
                    # Debug: show buffer status
                    if len(buffer) > 0:
                        print(f"📦 Buffer from {addr}: {len(buffer)} bytes waiting for complete message")
                        # Show more detailed buffer info for large buffers
                        if len(buffer) > 100000:  # If buffer is very large
                            print(f"   🔍 Large buffer detected - checking for message structure...")
                            if b'"type"' in buffer and b'"data_type"' in buffer:
                                print(f"   ✅ Buffer contains valid message structure, waiting for completion...")
                            else:
                                print(f"   ⚠️  Buffer may be corrupted or incomplete")
                        
                except socket.timeout:
                    # Check if we've been waiting too long for incomplete messages
                    if len(buffer) > 0 and (time.time() - last_activity) > 30:  # Increased timeout for large messages
                        print(f"⚠️  Timeout waiting for complete message from {addr}, clearing buffer")
                        print(f"   Buffer content: {buffer[:200]}...")
                        buffer = b""
                    continue
                    
        except Exception as e:
            print(f"❌ Data connection error from {addr}: {e}")
        finally:
            try:
                client_socket.close()
                print(f"🔌 Closed connection from {addr}")
            except:
                pass

    def process_pushed_data(self, message, addr):
        """Process data pushed from nodes"""
        try:
            msg_type = message.get("type")
            if msg_type == "data_push":
                self.message_count += 1
                data_type = message.get("data_type")
                node_name = message.get("node_name")
                data = message.get("data")
                timestamp = message.get("timestamp", time.time())
                
                print(f"📥 Message #{self.message_count}: Received {data_type} from {node_name} at {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
                print(f"   📊 Data type: {type(data)}, Size: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                if isinstance(data, str) and len(data) > 100:
                    print(f"   📝 Data preview: {data[:100]}...")
                elif isinstance(data, (list, tuple)) and len(data) > 10:
                    print(f"   📝 Data preview: {data[:10]}...")
                else:
                    print(f"   📝 Data: {data}")
                
                if data_type == "iq_samples":
                    # Convert string representation back to numpy array
                    try:
                        if isinstance(data, str):
                            iq_data = np.array(eval(data), dtype=np.complex64)
                        else:
                            iq_data = np.array(data, dtype=np.complex64)
                            
                        if node_name == "Alice":
                            self.alice_iq_data = iq_data
                            print(f"✅ Alice IQ samples: {len(iq_data)} samples")
                        elif node_name == "Bob":
                            self.bob_iq_data = iq_data
                            print(f"✅ Bob IQ samples: {len(iq_data)} samples")
                        
                        # Update visualization
                        if self.visualizer:
                            try:
                                self.visualizer.update_iq_data(node_name, iq_data)
                                print(f"🎨 Updated {node_name} IQ visualization - Data shape: {iq_data.shape}")
                            except Exception as e:
                                print(f"❌ Failed to update {node_name} IQ visualization: {e}")
                        else:
                            print(f"⚠️  No visualizer available for {node_name} IQ data")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} IQ samples: {e}")
                        print(f"   Data type: {type(data)}, Data: {str(data)[:100]}...")
                
                elif data_type == "spectrogram":
                    # Convert back to numpy array and reshape if needed
                    try:
                        if isinstance(data, str):
                            spec_data = np.array(eval(data), dtype=np.float32)
                        else:
                            spec_data = np.array(data, dtype=np.float32)

                        # Accept either 2D (204x31) or flattened length 6324
                        if spec_data.ndim == 1 and spec_data.size == 204*31:
                            spec_data = spec_data.reshape((204, 31))
                        elif spec_data.ndim != 2:
                            print(f"⚠️  Unexpected spectrogram shape from {node_name}: {spec_data.shape}")
                            
                        if node_name == "Alice":
                            self.alice_spectrogram_data = spec_data
                            print(f"✅ Alice spectrogram: shape {spec_data.shape}")
                        elif node_name == "Bob":
                            self.bob_spectrogram_data = spec_data
                            print(f"✅ Bob spectrogram: shape {spec_data.shape}")
                        
                        # Update visualization
                        if self.visualizer:
                            try:
                                self.visualizer.update_spectrogram(node_name, spec_data)
                                print(f"🎨 Updated {node_name} spectrogram visualization - Data shape: {spec_data.shape}")
                            except Exception as e:
                                print(f"❌ Failed to update {node_name} spectrogram visualization: {e}")
                        else:
                            print(f"⚠️  No visualizer available for {node_name} spectrogram data")
                    except Exception as e:
                        print(f"❌ Error processing {node_name} spectrogram: {e}")
                        print(f"   Data type: {type(data)}, Data: {str(data)[:100]}...")
                
                elif data_type == "quantized_bits":
                    print(f"🔍 Processing quantized bits from {node_name}")
                    print(f"   📊 Raw data type: {type(data)}")
                    print(f"   📊 Raw data: {str(data)[:100]}...")
                    
                    # Convert string representation back to bytes for BDR calculation
                    try:
                        if isinstance(data, list): # Data is now str(list) from control_layer, so it will be a list after json.loads
                            qb_data = bytes(data)
                            print(f"   ✅ Converted list to bytes: {len(qb_data)} bytes")
                        elif isinstance(data, str):
                            # Try to evaluate the string representation
                            try:
                                data_list = eval(data)
                                qb_data = bytes(data_list)
                                print(f"   ✅ Converted string to bytes: {len(qb_data)} bytes")
                            except Exception as e:
                                print(f"   ❌ Failed to convert string to bytes: {e}")
                                qb_data = None
                        else:
                            qb_data = data
                            print(f"   ⚠️  Using data as-is: {type(qb_data)}")
                        
                        if qb_data is not None:
                            if node_name == "Alice":
                                self.alice_quantized_bits = qb_data
                                print(f"✅ Alice quantized bits: {len(qb_data)} bytes")
                            elif node_name == "Bob":
                                self.bob_quantized_bits = qb_data
                                print(f"✅ Bob quantized bits: {len(qb_data)} bytes")
                            
                            # Check if we can calculate BDR
                            self.calculate_bdr_if_ready()
                        else:
                            print(f"❌ Failed to process quantized bits from {node_name}")
                            
                    except Exception as e:
                        print(f"❌ Error processing quantized bits from {node_name}: {e}")
                        print(f"   Data: {str(data)[:200]}...")
                
                elif data_type == "statistics":
                    # Store statistics
                    if node_name == "Alice":
                        self.alice_stats = data
                        print(f"✅ Alice statistics: Run {data.get('run_number')}, BDR: {data.get('bdr', 'N/A'):.4f}")
                    elif node_name == "Bob":
                        self.bob_stats = data
                        print(f"✅ Bob statistics: Run {data.get('run_number')}, BDR: {data.get('bdr', 'N/A'):.4f}")
                    
                    # Do not add a point yet; wait for timing to be computed to avoid duplicates
                    if data.get('bdr') is not None:
                        self.pending_bdr = float(data.get('bdr'))
                        print(f"⏳ Stored pending BDR from {node_name}: {self.pending_bdr:.4f}")
                
                elif data_type == "protocol_step":
                    # Normalize and update protocol step tracking
                    step_raw = str(data.get("step") or "")
                    normalized = self._normalize_protocol_step(node_name, step_raw)
                    if normalized is None:
                        print(f"⚠️  Ignored non-informative step from {node_name}: {data.get('step')}")
                    else:
                        # Derive base token without node prefix for timing/state logic
                        step_base = step_raw
                        if step_base.startswith("alice_"):
                            step_base = step_base[len("alice_"):]
                        elif step_base.startswith("bob_"):
                            step_base = step_base[len("bob_"):]

                        # Timing capture: mark start on Bob tx_start
                        now_ts = time.time()
                        if node_name == "Bob" and step_base == "tx_start" and self.timing['start_ts'] is None:
                            self.timing['start_ts'] = now_ts
                            print(f"⏱️  Timing start recorded at {now_ts:.3f}")
                        # Consider run "done" when reconciliation finishes (ok or fail)
                        if step_base in ("reconcile_ok", "reconcile_fail") or normalized == "Complete":
                            if node_name == "Alice":
                                self.timing['alice_done_ts'] = now_ts
                            elif node_name == "Bob":
                                self.timing['bob_done_ts'] = now_ts
                            # If both done, compute elapsed
                            if self.timing['start_ts'] and self.timing['alice_done_ts'] and self.timing['bob_done_ts']:
                                elapsed = max(self.timing['alice_done_ts'], self.timing['bob_done_ts']) - self.timing['start_ts']
                                print(f"⏱️  Elapsed time: {elapsed*1000:.1f} ms")
                                if self.visualizer:
                                    try:
                                        # Prefer pending BDR computed from quantized bits; fallback to stored stats; else 0.0
                                        bdr_value = self.pending_bdr if self.pending_bdr is not None else 0.0
                                        if bdr_value == 0.0 and self.alice_stats and isinstance(self.alice_stats.get('bdr'), (int, float)):
                                            bdr_value = float(self.alice_stats.get('bdr'))
                                        self.visualizer.add_run_statistics(bdr_value, True, int(elapsed*1000))
                                        print(f"🎨 Updated timing in visualization")
                                    except Exception as e:
                                        print(f"⚠️  Visualization timing update error: {e}")
                                # Reset timing for next run
                                self.timing = {'start_ts': None, 'alice_done_ts': None, 'bob_done_ts': None}
                                self.pending_bdr = None

                                # Auto-trigger next run shortly after completion
                                if self.auto_trigger:
                                    self._send_trigger_start(delay_s=0.5)
                        if node_name == "Alice":
                            self.alice_protocol_step = normalized
                            print(f"🔄 Alice protocol step: {self.alice_protocol_step}")
                        elif node_name == "Bob":
                            self.bob_protocol_step = normalized
                            print(f"🔄 Bob protocol step: {self.bob_protocol_step}")
                    
                    # Update visualization with protocol steps
                    if self.visualizer:
                        self.visualizer.update_step("Alice", self.alice_protocol_step)
                        self.visualizer.update_step("Bob", self.bob_protocol_step)
                        # Mark that visualization needs update (don't call from thread)
                        print(f"🎨 Updated protocol step visualization")
                
                elif data_type == "run_update":
                    # Update run tracking
                    run_number = data.get("run_number", 0)
                    action = data.get("action", "unknown")
                    if run_number > self.current_run:
                        self.current_run = run_number
                        print(f"🔄 {node_name} started run #{run_number}")
                    
                    # Update visualization with run number
                    if self.visualizer:
                        # Mark that visualization needs update (don't call from thread)
                        print(f"🎨 Updated run visualization")
                
        except Exception as e:
            print(f"❌ Error processing pushed data: {e}")

    def calculate_bdr_if_ready(self):
        """Calculate BDR when both nodes have quantized bits"""
        print(f"🔍 Checking if BDR can be calculated...")
        print(f"   Alice quantized bits: {'✅ Available' if self.alice_quantized_bits is not None else '❌ Not available'}")
        print(f"   Bob quantized bits: {'✅ Available' if self.bob_quantized_bits is not None else '❌ Not available'}")
        
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
                
                # Store BDR to be emitted together with timing to avoid duplicate points
                self.pending_bdr = float(bdr)
                print(f"💾 Stored pending BDR for next timing update: {self.pending_bdr:.4f}")
                
                # Clear the bits to avoid recalculating
                self.alice_quantized_bits = None
                self.bob_quantized_bits = None
                
            except Exception as e:
                print(f"❌ Error calculating BDR: {e}")

    def show_status(self):
        """Display current data status"""
        print("\n" + "="*70)
        print(f"🔐 Simple Push-Only PHYSEC Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        print(f"\n🚀 Protocol Status:")
        print(f"   Current Run: {self.current_run}")
        print(f"   Alice Protocol Step: {self.alice_protocol_step}")
        print(f"   Bob Protocol Step: {self.bob_protocol_step}")
        
        print(f"\n📊 Data Status:")
        print(f"   Alice IQ Samples: {'✅ Available' if self.alice_iq_data is not None else '❌ Not available'}")
        print(f"   Alice Spectrogram: {'✅ Available' if self.alice_spectrogram_data is not None else '❌ Not available'}")
        print(f"   Alice Quantized Bits: {'✅ Available' if self.alice_quantized_bits is not None else '❌ Not available'}")
        print(f"   Bob IQ Samples: {'✅ Available' if self.bob_iq_data is not None else '❌ Not available'}")
        print(f"   Bob Spectrogram: {'✅ Available' if self.bob_spectrogram_data is not None else '❌ Not available'}")
        print(f"   Bob Quantized Bits: {'✅ Available' if self.bob_quantized_bits is not None else '❌ Not available'}")
        
        if self.alice_stats:
            print(f"\n📈 Alice Statistics:")
            print(f"   Run: {self.alice_stats.get('run_number', 'N/A')}")
            print(f"   BDR: {self.alice_stats.get('bdr', 'N/A'):.4f}" if self.alice_stats.get('bdr') is not None else "   BDR: N/A")
            print(f"   Success: {'✅' if self.alice_stats.get('success') else '❌'}")
        
        if self.bob_stats:
            print(f"\n📈 Bob Statistics:")
            print(f"   Run: {self.bob_stats.get('run_number', 'N/A')}")
            print(f"   BDR: {self.bob_stats.get('bdr', 'N/A'):.4f}" if self.bob_stats.get('bdr') is not None else "   BDR: N/A")
            print(f"   Success: {'✅' if self.bob_stats.get('success') else '❌'}")
        
        print("="*70)

    def start_monitoring(self):
        """Start monitoring with data collection server"""
        print(f"\n🔄 Simple push-only monitoring started... (Press Ctrl+C to stop)")
        print(f"💡 Data will be pushed automatically from nodes")
        
        # Start data collection server in background
        server_thread = threading.Thread(target=self.start_data_collection_server, daemon=True)
        server_thread.start()
        
        # Initialize trigger publisher and send initial start after a brief delay
        if self.auto_trigger:
            self._start_trigger_publisher()
            self._send_trigger_start(delay_s=1.0)
        
        try:
            while self.data_server_running:
                # Update visualization more frequently for smooth updates
                time.sleep(0.1)
                
                # Update visualization in main thread
                if self.visualizer and self.visualizer.running:
                    try:
                        self.visualizer.update_display()
                    except Exception as e:
                        print(f"⚠️  Visualization update error: {e}")
                
                # Periodically check quantized bits status
                if int(time.time()) % 10 == 0:  # Every 10 seconds
                    if self.alice_quantized_bits is None or self.bob_quantized_bits is None:
                        print(f"⏳ Waiting for quantized bits... Alice: {'✅' if self.alice_quantized_bits is not None else '❌'}, Bob: {'✅' if self.bob_quantized_bits is not None else '❌'}")
                

                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping simple push-only monitor...")
        finally:
            self.data_server_running = False
            if self.data_server_socket:
                self.data_server_socket.close()
            
            # Show final summary
            print(f"\n📊 Final Summary:")
            print(f"   Total messages received: {self.message_count}")
            print(f"   Alice IQ data: {'✅' if self.alice_iq_data is not None else '❌'}")
            print(f"   Alice spectrogram: {'✅' if self.alice_spectrogram_data is not None else '❌'}")
            print(f"   Bob IQ data: {'✅' if self.bob_iq_data is not None else '❌'}")
            print(f"   Bob spectrogram: {'✅' if self.bob_spectrogram_data is not None else '❌'}")
            print(f"   Alice quantized bits: {'✅' if self.alice_quantized_bits is not None else '❌'}")
            print(f"   Bob quantized bits: {'✅' if self.bob_quantized_bits is not None else '❌'}")
            print(f"   BDR calculation: {'✅' if (self.alice_quantized_bits is not None and self.bob_quantized_bits is not None) else '❌'}")
            
            if self.visualizer:
                try:
                    self.visualizer.stop_visualization()
                    print("🎨 Visualization stopped")
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description="Simple Push-Only PHYSEC Protocol Monitor")
    parser.add_argument("--alice-ip", required=True, help="Alice's IP address")
    parser.add_argument("--bob-ip", required=True, help="Bob's IP address")
    
    args = parser.parse_args()
    
    monitor = SimplePushMonitor(args.alice_ip, args.bob_ip)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
