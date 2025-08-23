#!/usr/bin/env python3
"""
Simple PHYSEC Protocol Monitor - Receives pushed data from nodes
"""

import socket
import json
import time
import argparse
import sys
import os
import numpy as np
import threading

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dynamic_visualization import PhysecDynamicVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import visualization: {e}")
    VISUALIZATION_AVAILABLE = False

class SimpleProtocolMonitor:
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
        
        # Data storage
        self.alice_iq_data = None
        self.alice_spectrogram_data = None
        self.alice_quantized_bits = None
        self.bob_iq_data = None
        self.bob_spectrogram_data = None
        self.bob_quantized_bits = None
        
        # Statistics storage
        self.alice_stats = {}
        self.bob_stats = {}
        
        # Server for receiving pushed data
        self.server_socket = None
        self.running = True
        
        print(f"🔧 Starting Simple PHYSEC Protocol Monitor...")
        print(f"📡 Will receive data from Alice: {alice_ip}:{alice_port}")
        print(f"📡 Will receive data from Bob:   {bob_ip}:{bob_port}")
        
        if self.visualizer:
            print("🎨 Dynamic visualization enabled")
        else:
            print("⚠️  Visualization disabled - text-only monitoring")
        
        print("⏳ Starting data collection server...")

    def start_data_server(self):
        """Start server to receive pushed data from nodes"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', 9999))  # Use port 9999 for data collection
            self.server_socket.listen(5)
            
            print(f"✅ Data collection server listening on port 9999")
            
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"📡 Data connection from {addr}")
                    threading.Thread(target=self.handle_data_connection, args=(client_socket, addr)).start()
                except Exception as e:
                    if self.running:
                        print(f"❌ Server error: {e}")
                        
        except Exception as e:
            print(f"❌ Failed to start data server: {e}")

    def handle_data_connection(self, client_socket, addr):
        """Handle data connection from a node"""
        try:
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8').strip())
                    self.process_pushed_data(message, addr)
                except json.JSONDecodeError:
                    # Handle partial messages
                    continue
                    
        except Exception as e:
            print(f"❌ Data connection error from {addr}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass

    def process_pushed_data(self, message, addr):
        """Process data pushed from nodes"""
        try:
            msg_type = message.get("type")
            if msg_type == "data_push":
                data_type = message.get("data_type")
                node_name = message.get("node_name")
                data = message.get("data")
                timestamp = message.get("timestamp", time.time())
                
                print(f"📥 Received {data_type} from {node_name} at {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
                
                if data_type == "iq_samples":
                    # Convert string representation back to numpy array
                    iq_data = np.array(eval(data), dtype=np.complex64)
                    if node_name == "Alice":
                        self.alice_iq_data = iq_data
                        print(f"✅ Alice IQ samples: {len(iq_data)} samples")
                    elif node_name == "Bob":
                        self.bob_iq_data = iq_data
                        print(f"✅ Bob IQ samples: {len(iq_data)} samples")
                    
                    # Update visualization
                    if self.visualizer:
                        self.visualizer.update_iq_data(node_name, iq_data)
                        self.visualizer.process_events()
                        self.visualizer.update_display()
                
                elif data_type == "spectrogram":
                    # Convert string representation back to numpy array
                    spec_data = np.array(eval(data), dtype=np.float32)
                    if node_name == "Alice":
                        self.alice_spectrogram_data = spec_data
                        print(f"✅ Alice spectrogram: shape {spec_data.shape}")
                    elif node_name == "Bob":
                        self.bob_spectrogram_data = spec_data
                        print(f"✅ Bob spectrogram: shape {spec_data.shape}")
                    
                    # Update visualization
                    if self.visualizer:
                        self.visualizer.update_spectrogram(node_name, spec_data)
                        self.visualizer.process_events()
                        self.visualizer.update_display()
                
                elif data_type == "quantized_bits":
                    # Convert string representation back to bytes
                    qb_data = bytes(eval(data))
                    if node_name == "Alice":
                        self.alice_quantized_bits = qb_data
                        print(f"✅ Alice quantized bits: {len(qb_data)} bytes")
                    elif node_name == "Bob":
                        self.bob_quantized_bits = qb_data
                        print(f"✅ Bob quantized bits: {len(qb_data)} bytes")
                    
                    # Check if we can calculate BDR
                    self.calculate_bdr_if_ready()
                
                elif data_type == "statistics":
                    # Store statistics
                    if node_name == "Alice":
                        self.alice_stats = data
                        print(f"✅ Alice statistics: Run {data.get('run_number')}, BDR: {data.get('bdr', 'N/A'):.4f}")
                    elif node_name == "Bob":
                        self.bob_stats = data
                        print(f"✅ Bob statistics: Run {data.get('run_number')}, BDR: {data.get('bdr', 'N/A'):.4f}")
                    
                    # Update visualization with statistics
                    if self.visualizer and data.get('bdr') is not None:
                        self.visualizer.add_run_statistics(
                            data.get('bdr'),
                            data.get('success', True),
                            data.get('timing_ms')
                        )
                        self.visualizer.process_events()
                        self.visualizer.update_display()
                
        except Exception as e:
            print(f"❌ Error processing pushed data: {e}")

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
                    # Use placeholder values for now
                    success = True
                    timing_ms = None
                    
                    self.visualizer.add_run_statistics(bdr, success, timing_ms)
                    self.visualizer.process_events()
                    self.visualizer.update_display()
                
                # Clear the bits to avoid recalculating
                self.alice_quantized_bits = None
                self.bob_quantized_bits = None
                
            except Exception as e:
                print(f"❌ Error calculating BDR: {e}")

    def show_status(self):
        """Display current data status"""
        print("\n" + "="*70)
        print(f"🔐 Simple PHYSEC Protocol Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
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
        print(f"\n🔄 Simple monitoring started... (Press Ctrl+C to stop)")
        print(f"💡 Data will be pushed automatically from nodes")
        
        # Start data collection server in background
        server_thread = threading.Thread(target=self.start_data_server, daemon=True)
        server_thread.start()
        
        try:
            while self.running:
                # Show status every 10 seconds
                time.sleep(10)
                self.show_status()
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping simple protocol monitor...")
        finally:
            self.running = False
            if self.server_socket:
                self.server_socket.close()
            if self.visualizer:
                try:
                    self.visualizer.stop_visualization()
                    print("🎨 Visualization stopped")
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description="Simple PHYSEC Protocol Monitor - Receives pushed data")
    parser.add_argument("--alice-ip", required=True, help="Alice's IP address")
    parser.add_argument("--bob-ip", required=True, help="Bob's IP address")
    parser.add_argument("--alice-port", type=int, default=9001, help="Alice's monitoring port (default: 9001)")
    parser.add_argument("--bob-port", type=int, default=9002, help="Bob's monitoring port (default: 9002)")
    
    args = parser.parse_args()
    
    monitor = SimpleProtocolMonitor(args.alice_ip, args.bob_ip, args.alice_port, args.bob_port)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
