#!/usr/bin/env python3
"""
Network-aware PHYSEC Demo
Supports distributed Alice/Bob nodes with remote visualization
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import socket
import json
import threading
import time
import argparse
from dynamic_visualization import DynamicVisualization

class NetworkMonitor:
    """Monitor PHYSEC protocol execution across network"""
    
    def __init__(self, alice_host, alice_port, bob_host, bob_port):
        self.alice_host = alice_host
        self.alice_port = alice_port
        self.bob_host = bob_host  
        self.bob_port = bob_port
        self.visualizer = None
        self.running = True
        
    def start_monitoring(self):
        """Start monitoring both Alice and Bob nodes"""
        print("🌐 Starting network monitoring...")
        print(f"   Alice: {self.alice_host}:{self.alice_port}")
        print(f"   Bob: {self.bob_host}:{self.bob_port}")
        
        # Start visualization
        self.visualizer = DynamicVisualization(max_runs=10)
        self.visualizer.start_visualization()
        
        # Start monitoring threads
        alice_thread = threading.Thread(target=self._monitor_node, 
                                       args=("Alice", self.alice_host, self.alice_port))
        bob_thread = threading.Thread(target=self._monitor_node,
                                     args=("Bob", self.bob_host, self.bob_port))
        
        alice_thread.daemon = True
        bob_thread.daemon = True
        alice_thread.start()
        bob_thread.start()
        
        try:
            # Keep visualization alive
            while self.running:
                if self.visualizer:
                    self.visualizer.process_events()
                    self.visualizer.update_display()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            self.running = False
            
        if self.visualizer:
            self.visualizer.stop_visualization()
    
    def _monitor_node(self, node_name, host, port):
        """Monitor a specific node via TCP connection"""
        while self.running:
            try:
                # Connect to node's monitoring port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((host, port + 100))  # Monitoring port = main port + 100
                
                print(f"✅ Connected to {node_name} monitoring at {host}:{port + 100}")
                
                while self.running:
                    # Receive monitoring data
                    data = sock.recv(4096)
                    if not data:
                        break
                        
                    try:
                        message = json.loads(data.decode())
                        self._process_monitoring_message(node_name, message)
                    except json.JSONDecodeError:
                        continue
                        
            except Exception as e:
                print(f"❌ {node_name} monitoring error: {e}")
                time.sleep(5.0)  # Retry after 5 seconds
            finally:
                try:
                    sock.close()
                except:
                    pass
    
    def _process_monitoring_message(self, node_name, message):
        """Process monitoring message from a node"""
        msg_type = message.get("type")
        
        if msg_type == "step_update":
            step = message.get("step")
            if self.visualizer:
                self.visualizer.update_step(f"{node_name}: {step}")
                
        elif msg_type == "iq_data":
            iq_samples = message.get("samples")
            if self.visualizer and iq_samples:
                if node_name == "Alice":
                    self.visualizer.update_iq_alice(iq_samples)
                else:
                    self.visualizer.update_iq_bob(iq_samples)
                    
        elif msg_type == "spectrogram_data":
            spec_data = message.get("spectrogram")
            if self.visualizer and spec_data:
                if node_name == "Alice":
                    self.visualizer.update_spectrogram_alice(spec_data)
                else:
                    self.visualizer.update_spectrogram_bob(spec_data)
                    
        elif msg_type == "run_statistics":
            bdr = message.get("bdr", 0)
            success = message.get("success", False)
            timing_ms = message.get("timing_ms", 0)
            if self.visualizer:
                self.visualizer.add_run_statistics(bdr, success, timing_ms)

def main():
    parser = argparse.ArgumentParser(description='PHYSEC Network Monitor')
    parser.add_argument('--alice-host', default='localhost',
                       help='Alice node hostname/IP')
    parser.add_argument('--alice-port', type=int, default=8001,
                       help='Alice node port')
    parser.add_argument('--bob-host', default='localhost', 
                       help='Bob node hostname/IP')
    parser.add_argument('--bob-port', type=int, default=8002,
                       help='Bob node port')
    
    args = parser.parse_args()
    
    monitor = NetworkMonitor(args.alice_host, args.alice_port,
                           args.bob_host, args.bob_port)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
