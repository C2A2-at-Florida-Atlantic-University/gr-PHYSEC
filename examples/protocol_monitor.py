#!/usr/bin/env python3
"""
Protocol Monitor for Distributed PHYSEC Deployment
Connects to running Alice and Bob nodes to monitor protocol execution in real-time
"""

import socket
import json
import time
import threading
import argparse
from datetime import datetime
import sys
import os

# Add the current directory to Python path to import dynamic_visualization
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dynamic_visualization import PhysecDynamicVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import dynamic_visualization: {e}")
    print("   The monitor will run in text-only mode")
    VISUALIZATION_AVAILABLE = False

class PHYSECProtocolMonitor:
    """Monitor PHYSEC protocol execution in real-time"""
    
    def __init__(self, alice_ip, bob_ip, alice_port=8001, bob_port=8002):
        self.alice_ip = alice_ip
        self.bob_ip = bob_ip
        self.alice_port = alice_port
        self.bob_port = bob_port
        
        # Protocol state tracking
        self.alice_state = "unknown"
        self.bob_state = "unknown"
        self.current_run = 0
        self.alice_run_state = "idle"
        self.bob_run_state = "idle"
        
        # Statistics
        self.successful_runs = 0
        self.failed_runs = 0
        self.run_durations = []
        
        # Visualization
        self.visualizer = None
        if VISUALIZATION_AVAILABLE:
            try:
                self.visualizer = PhysecDynamicVisualizer()
                self.visualizer.start_visualization()
                print("✅ Dynamic visualization started")
            except Exception as e:
                print(f"⚠️  Could not start visualization: {e}")
                self.visualizer = None
        
        self.running = True
        
    def connect_to_node(self, ip, port, node_name):
        """Connect to a node and get its current state"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((ip, port))
            
            # Send a status request
            request = {
                "type": "status_request",
                "timestamp": time.time()
            }
            sock.send(json.dumps(request).encode('utf-8') + b'\n')
            
            # Wait for response
            sock.settimeout(5.0)
            data = sock.recv(1024)
            sock.close()
            
            if data:
                try:
                    response = json.loads(data.decode('utf-8').strip())
                    return response
                except json.JSONDecodeError:
                    return {"state": "unknown", "error": "Invalid JSON response"}
            else:
                return {"state": "unknown", "error": "No response"}
                
        except Exception as e:
            # If connection fails, the node might be busy with another connection
            # Try a different approach - just check if the port is open
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.settimeout(1.0)
                result = test_sock.connect_ex((ip, port))
                test_sock.close()
                
                if result == 0:
                    # Port is open but we can't establish a new connection
                    # This usually means the node is busy with another session
                    return {"state": "busy", "error": "Node is busy with another connection"}
                else:
                    return {"state": "disconnected", "error": str(e)}
            except:
                return {"state": "disconnected", "error": str(e)}
    
    def monitor_alice_protocol(self):
        """Monitor Alice's protocol state"""
        while self.running:
            try:
                response = self.connect_to_node(self.alice_ip, self.alice_port, "Alice")
                old_state = self.alice_state
                self.alice_state = response.get("state", "unknown")
                
                if self.alice_state != old_state:
                    print(f"🔄 Alice state changed: {old_state} → {self.alice_state}")
                    if self.visualizer:
                        self.visualizer.update_step("Alice", self.alice_state)
                
                # Extract run information if available
                if "run_number" in response:
                    self.current_run = response["run_number"]
                if "run_state" in response:
                    self.alice_run_state = response["run_state"]
                    
            except Exception as e:
                print(f"❌ Error monitoring Alice: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def monitor_bob_protocol(self):
        """Monitor Bob's protocol state"""
        while self.running:
            try:
                response = self.connect_to_node(self.bob_ip, self.bob_port, "Bob")
                old_state = self.bob_state
                self.bob_state = response.get("state", "unknown")
                
                if self.bob_state != old_state:
                    print(f"🔄 Bob state changed: {old_state} → {self.bob_state}")
                    if self.visualizer:
                        self.visualizer.update_step("Bob", self.bob_state)
                
                # Extract run information if available
                if "run_number" in response:
                    self.current_run = response["run_number"]
                if "run_state" in response:
                    self.bob_run_state = response["run_state"]
                    
            except Exception as e:
                print(f"❌ Error monitoring Bob: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def show_protocol_status(self):
        """Display current protocol status"""
        print("\n" + "="*70)
        print(f"🔐 PHYSEC Protocol Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Node states
        print(f"\n📱 Node States:")
        print(f"   Alice ({self.alice_ip}:{self.alice_port}): {self.alice_state}")
        print(f"   Bob   ({self.bob_ip}:{self.bob_port}): {self.bob_state}")
        
        # Protocol execution
        print(f"\n🚀 Protocol Execution:")
        print(f"   Current Run: {self.current_run}")
        print(f"   Alice Run State: {self.alice_run_state}")
        print(f"   Bob Run State: {self.bob_run_state}")
        
        # Statistics
        print(f"\n📊 Statistics:")
        print(f"   Successful Runs: {self.successful_runs}")
        print(f"   Failed Runs: {self.failed_runs}")
        if self.run_durations:
            avg_duration = sum(self.run_durations) / len(self.run_durations)
            print(f"   Average Duration: {avg_duration:.2f}ms")
        
        # Protocol progress
        print(f"\n🎯 Protocol Progress:")
        if self.alice_state == "key_ready" and self.bob_state == "key_ready":
            print("   ✅ Both nodes completed key generation successfully!")
        elif self.alice_state == "error" or self.bob_state == "error":
            print("   ❌ One or both nodes encountered an error")
        elif self.alice_state == "reconciliation_failed" or self.bob_state == "reconciliation_failed":
            print("   ⚠️  Reconciliation failed on one or both nodes")
        elif self.alice_state in ["processing", "sent_parity"] or self.bob_state in ["processing", "sent_parity"]:
            print("   🔄 Protocol in progress - processing samples")
        elif self.alice_state == "transmitting" or self.bob_state == "transmitting":
            print("   📡 Probe transmission in progress")
        elif self.alice_state == "requesting" or self.bob_state == "accepted":
            print("   🤝 Key generation request initiated")
        elif self.alice_state == "busy" or self.bob_state == "busy":
            print("   🔒 One or both nodes are busy with another connection")
            print("   💡 This is normal when Alice and Bob are actively communicating")
        else:
            print("   ⏸️  Protocol idle or unknown state")
        
        print("="*70)
    
    def start_monitoring(self):
        """Start protocol monitoring"""
        print("🔧 Starting PHYSEC Protocol Monitor...")
        print(f"📡 Monitoring Alice: {self.alice_ip}:{self.alice_port}")
        print(f"📡 Monitoring Bob:   {self.bob_ip}:{self.bob_port}")
        
        if self.visualizer:
            print("🎨 Dynamic visualization enabled")
            print("   • Real-time protocol step updates")
            print("   • IQ data and spectrogram visualization")
            print("   • Success rate and timing statistics")
        else:
            print("📝 Running in text-only mode")
            
        print("⏳ Starting protocol monitoring...")
        
        # Start monitoring threads
        alice_thread = threading.Thread(target=self.monitor_alice_protocol, daemon=True)
        bob_thread = threading.Thread(target=self.monitor_bob_protocol, daemon=True)
        
        alice_thread.start()
        bob_thread.start()
        
        print(f"\n🔄 Protocol monitoring started... (Press Ctrl+C to stop)")
        
        try:
            while self.running:
                time.sleep(15)  # Update status every 15 seconds
                self.show_protocol_status()
                
                # Update visualization if available
                if self.visualizer:
                    self.visualizer.process_events()
                    self.visualizer.update_display()
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping protocol monitor...")
            self.running = False
            
            # Stop visualization
            if self.visualizer:
                self.visualizer.stop_visualization()
                print("🎨 Visualization stopped")

def main():
    parser = argparse.ArgumentParser(description='PHYSEC Protocol Monitor')
    parser.add_argument('--alice-ip', required=True, help='Alice node IP address')
    parser.add_argument('--bob-ip', required=True, help='Bob node IP address')
    parser.add_argument('--alice-port', type=int, default=8001, help='Alice port (default: 8001)')
    parser.add_argument('--bob-port', type=int, default=8002, help='Bob port (default: 8002)')
    
    args = parser.parse_args()
    
    print("🔐 PHYSEC Protocol Monitor")
    print("=" * 50)
    
    monitor = PHYSECProtocolMonitor(
        alice_ip=args.alice_ip,
        bob_ip=args.bob_ip,
        alice_port=args.alice_port,
        bob_port=args.bob_port
    )
    
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
