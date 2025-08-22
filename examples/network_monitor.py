#!/usr/bin/env python3
"""
Enhanced Network Monitor for Distributed PHYSEC Deployment
Uses dynamic visualization to monitor Alice and Bob nodes in real-time
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

class PHYSECNetworkMonitor:
    """Enhanced monitor with dynamic visualization for distributed PHYSEC deployment"""
    
    def __init__(self, alice_ip, bob_ip, alice_port=8001, bob_port=8002):
        self.alice_ip = alice_ip
        self.bob_ip = bob_ip
        self.alice_port = alice_port
        self.bob_port = bob_port
        
        # Network status
        self.alice_status = "disconnected"
        self.bob_status = "disconnected"
        self.alice_messages = []
        self.bob_messages = []
        
        # PHYSEC protocol tracking
        self.alice_protocol_state = "idle"
        self.bob_protocol_state = "idle"
        self.current_run = 0
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        
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
        
    def test_connection(self, host, port, node_name):
        """Test if a node is reachable"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return "connected"
            else:
                return "port_closed"
        except Exception as e:
            return f"error: {e}"
    
    def monitor_alice(self):
        """Monitor Alice's network status"""
        while self.running:
            status = self.test_connection(self.alice_ip, self.alice_port, "Alice")
            old_status = self.alice_status
            self.alice_status = status
            
            if status != old_status:
                if status == "connected":
                    print(f"✅ Alice ({self.alice_ip}:{self.alice_port}) - {status}")
                    if self.visualizer:
                        self.visualizer.update_step("Alice", "Connected")
                else:
                    print(f"❌ Alice ({self.alice_ip}:{self.alice_port}) - {status}")
                    if self.visualizer:
                        self.visualizer.update_step("Alice", "Disconnected")
            
            time.sleep(10)  # Check every 10 seconds
    
    def monitor_bob(self):
        """Monitor Bob's network status"""
        while self.running:
            status = self.test_connection(self.bob_ip, self.bob_port, "Bob")
            old_status = self.bob_status
            self.bob_status = status
            
            if status != old_status:
                if status == "connected":
                    print(f"✅ Bob ({self.bob_ip}:{self.bob_port}) - {status}")
                    if self.visualizer:
                        self.visualizer.update_step("Bob", "Connected")
                else:
                    print(f"❌ Bob ({self.bob_ip}:{self.bob_port}) - {status}")
                    if self.visualizer:
                        self.visualizer.update_step("Bob", "Disconnected")
            
            time.sleep(10)  # Check every 10 seconds
    
    def update_protocol_state(self, node_name, state, run_info=None):
        """Update protocol state for visualization"""
        if self.visualizer:
            self.visualizer.update_step(node_name, state)
            
            if run_info and 'run_number' in run_info:
                self.current_run = run_info['run_number']
                if 'success' in run_info:
                    if run_info['success']:
                        self.successful_runs += 1
                    else:
                        self.failed_runs += 1
                        
                    # Update statistics in visualization
                    self.visualizer.add_run_statistics(
                        success=run_info['success'],
                        duration_ms=run_info.get('duration_ms', 0)
                    )
    
    def show_status(self):
        """Display current status"""
        print("\n" + "="*60)
        print(f"🌐 PHYSEC Network Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Network status
        print(f"\n📡 Network Status:")
        print(f"   Alice ({self.alice_ip}:{self.alice_port}): {self.alice_status}")
        print(f"   Bob   ({self.bob_ip}:{self.bob_port}): {self.bob_status}")
        
        # Protocol status
        print(f"\n🔐 PHYSEC Protocol Status:")
        print(f"   Current Run: {self.current_run}")
        print(f"   Alice State: {self.alice_protocol_state}")
        print(f"   Bob State: {self.bob_protocol_state}")
        print(f"   Successful Runs: {self.successful_runs}")
        print(f"   Failed Runs: {self.failed_runs}")
        
        # Overall status
        if self.alice_status == "connected" and self.bob_status == "connected":
            print(f"\n✅ Both nodes are reachable - Ready for PHYSEC protocol!")
        elif self.alice_status == "connected":
            print(f"\n⚠️  Only Alice is reachable - Check Bob's network/firewall")
        elif self.bob_status == "connected":
            print(f"\n⚠️  Only Bob is reachable - Check Alice's network/firewall")
        else:
            print(f"\n❌ Neither node is reachable - Check network configuration")
        
        # Instructions
        print(f"\n📋 Next Steps:")
        if self.alice_status == "connected" and self.bob_status == "connected":
            print(f"   1. ✅ Network is ready!")
            print(f"   2. 🚀 Start Alice: python3 control_layer.py --node alice --peer-host {self.bob_ip} --runs 2")
            print(f"   3. 🚀 Start Bob:   python3 control_layer.py --node bob --peer-host {self.alice_ip} --runs 2")
            print(f"   4. 📊 Watch the protocol execution in real-time visualization")
        else:
            print(f"   1. 🔧 Fix network connectivity issues")
            print(f"   2. 🔥 Check firewalls on both Jetson Orins")
            print(f"   3. 🌐 Verify IP addresses and routing")
            print(f"   4. 🔄 Restart this monitor after fixes")
        
        print(f"\n💡 Tips:")
        print(f"   • Keep this monitor running to see real-time status")
        print(f"   • The visualization shows protocol steps, IQ data, and statistics")
        print(f"   • Use Ctrl+C to stop monitoring")
        print(f"   • Check both Jetson Orins for detailed logs")
    
    def start_monitoring(self):
        """Start enhanced monitoring with visualization"""
        print("🔧 Starting Enhanced PHYSEC Network Monitor...")
        print(f"📡 Monitoring Alice: {self.alice_ip}:{self.alice_port}")
        print(f"📡 Monitoring Bob:   {self.bob_ip}:{self.bob_port}")
        
        if self.visualizer:
            print("🎨 Dynamic visualization enabled")
        else:
            print("📝 Running in text-only mode")
            
        print("⏳ Initial status check...")
        
        # Initial status check
        self.alice_status = self.test_connection(self.alice_ip, self.alice_port, "Alice")
        self.bob_status = self.test_connection(self.bob_ip, self.bob_port, "Bob")
        
        # Show initial status
        self.show_status()
        
        # Start monitoring threads
        alice_thread = threading.Thread(target=self.monitor_alice, daemon=True)
        bob_thread = threading.Thread(target=self.monitor_bob, daemon=True)
        
        alice_thread.start()
        bob_thread.start()
        
        print(f"\n🔄 Continuous monitoring started... (Press Ctrl+C to stop)")
        
        try:
            while self.running:
                time.sleep(30)  # Update status every 30 seconds
                self.show_status()
                
                # Update visualization if available
                if self.visualizer:
                    self.visualizer.process_events()
                    self.visualizer.update_display()
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping monitor...")
            self.running = False
            
            # Stop visualization
            if self.visualizer:
                self.visualizer.stop_visualization()
                print("🎨 Visualization stopped")

def main():
    parser = argparse.ArgumentParser(description='Enhanced PHYSEC Network Monitor')
    parser.add_argument('--alice-ip', required=True, help='Alice node IP address')
    parser.add_argument('--bob-ip', required=True, help='Bob node IP address')
    parser.add_argument('--alice-port', type=int, default=8001, help='Alice port (default: 8001)')
    parser.add_argument('--bob-port', type=int, default=8002, help='Bob port (default: 8002)')
    
    args = parser.parse_args()
    
    print("🌐 Enhanced PHYSEC Network Monitor")
    print("=" * 50)
    
    monitor = PHYSECNetworkMonitor(
        alice_ip=args.alice_ip,
        bob_ip=args.bob_ip,
        alice_port=args.alice_port,
        bob_port=args.bob_port
    )
    
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
