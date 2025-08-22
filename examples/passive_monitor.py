#!/usr/bin/env python3
"""
Passive PHYSEC Protocol Monitor
Watches network traffic and provides real-time monitoring without interfering with active connections
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

class PassivePHYSECMonitor:
    """Passive monitor that watches network activity without interfering"""
    
    def __init__(self, alice_ip, bob_ip, alice_port=8001, bob_port=8002):
        self.alice_ip = alice_ip
        self.bob_ip = bob_ip
        self.alice_port = alice_port
        self.bob_port = bob_port
        
        # Protocol state tracking (inferred from network activity)
        self.alice_state = "unknown"
        self.bob_state = "unknown"
        self.current_run = 0
        self.last_activity = {}
        
        # Statistics
        self.successful_runs = 0
        self.failed_runs = 0
        self.total_messages = 0
        
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
        
    def check_port_activity(self, ip, port, node_name):
        """Check if a port is active (has recent activity)"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result == 0:
                # Port is open and accepting connections
                if node_name not in self.last_activity:
                    self.last_activity[node_name] = time.time()
                    print(f"✅ {node_name} port {port} is now active")
                    
                    # Update visualization
                    if self.visualizer:
                        self.visualizer.update_step(node_name, "Active")
                
                return "active"
            else:
                return "closed"
                
        except Exception as e:
            return "error"
    
    def infer_protocol_state(self, node_name, port_status):
        """Infer protocol state based on port activity and timing"""
        if port_status == "active":
            if node_name in self.last_activity:
                time_since_activity = time.time() - self.last_activity[node_name]
                
                if time_since_activity < 5:  # Very recent activity
                    return "processing"
                elif time_since_activity < 30:  # Recent activity
                    return "active"
                else:  # Stale activity
                    return "idle"
            else:
                return "unknown"
        else:
            return "disconnected"
    
    def monitor_alice(self):
        """Monitor Alice's network activity"""
        while self.running:
            try:
                port_status = self.check_port_activity(self.alice_ip, self.alice_port, "Alice")
                old_state = self.alice_state
                self.alice_state = self.infer_protocol_state("Alice", port_status)
                
                if self.alice_state != old_state:
                    print(f"🔄 Alice state inferred: {old_state} → {self.alice_state}")
                    if self.visualizer:
                        self.visualizer.update_step("Alice", self.alice_state)
                        
            except Exception as e:
                print(f"❌ Error monitoring Alice: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def monitor_bob(self):
        """Monitor Bob's network activity"""
        while self.running:
            try:
                port_status = self.check_port_activity(self.bob_ip, self.bob_port, "Bob")
                old_state = self.bob_state
                self.bob_state = self.infer_protocol_state("Bob", port_status)
                
                if self.bob_state != old_state:
                    print(f"🔄 Bob state inferred: {old_state} → {self.bob_state}")
                    if self.visualizer:
                        self.visualizer.update_step("Bob", self.bob_state)
                        
            except Exception as e:
                print(f"❌ Error monitoring Bob: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def show_monitoring_status(self):
        """Display current monitoring status"""
        print("\n" + "="*70)
        print(f"👁️  Passive PHYSEC Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Network activity
        print(f"\n📡 Network Activity:")
        print(f"   Alice ({self.alice_ip}:{self.alice_port}): {self.alice_state}")
        if "Alice" in self.last_activity:
            time_since = time.time() - self.last_activity["Alice"]
            print(f"      Last activity: {time_since:.1f}s ago")
            
        print(f"   Bob   ({self.bob_ip}:{self.bob_port}): {self.bob_state}")
        if "Bob" in self.last_activity:
            time_since = time.time() - self.last_activity["Bob"]
            print(f"      Last activity: {time_since:.1f}s ago")
        
        # Inferred protocol state
        print(f"\n🔍 Inferred Protocol State:")
        if self.alice_state == "active" and self.bob_state == "active":
            print("   🚀 Both nodes are actively communicating")
            print("   💡 PHYSEC protocol is likely in progress")
        elif self.alice_state == "processing" or self.bob_state == "processing":
            print("   🔄 One or both nodes are processing data")
            print("   💡 PHYSEC pipeline is active")
        elif self.alice_state == "idle" and self.bob_state == "idle":
            print("   ⏸️  Both nodes are idle")
            print("   💡 Protocol may be waiting or completed")
        elif self.alice_state == "disconnected" or self.bob_state == "disconnected":
            print("   ❌ One or both nodes are unreachable")
            print("   💡 Check network connectivity")
        
        # Monitoring info
        print(f"\n📊 Monitoring Information:")
        print(f"   Total Messages Detected: {self.total_messages}")
        print(f"   Successful Runs Inferred: {self.successful_runs}")
        print(f"   Failed Runs Inferred: {self.failed_runs}")
        
        # Instructions
        print(f"\n💡 How This Works:")
        print(f"   • This monitor watches network activity passively")
        print(f"   • It doesn't interfere with Alice/Bob communication")
        print(f"   • States are inferred from port activity and timing")
        print(f"   • For detailed logs, check the console on each node")
        
        print("="*70)
    
    def start_monitoring(self):
        """Start passive monitoring"""
        print("🔧 Starting Passive PHYSEC Monitor...")
        print(f"📡 Monitoring Alice: {self.alice_ip}:{self.alice_port}")
        print(f"📡 Monitoring Bob:   {self.bob_ip}:{self.bob_port}")
        
        if self.visualizer:
            print("🎨 Dynamic visualization enabled")
            print("   • Real-time state updates based on network activity")
            print("   • Protocol progress visualization")
            print("   • Success rate and timing statistics")
        else:
            print("📝 Running in text-only mode")
            
        print("⏳ Starting passive monitoring...")
        print("💡 This monitor won't interfere with active connections")
        
        # Start monitoring threads
        alice_thread = threading.Thread(target=self.monitor_alice, daemon=True)
        bob_thread = threading.Thread(target=self.monitor_bob, daemon=True)
        
        alice_thread.start()
        bob_thread.start()
        
        print(f"\n🔄 Passive monitoring started... (Press Ctrl+C to stop)")
        
        try:
            while self.running:
                time.sleep(20)  # Update status every 20 seconds
                self.show_monitoring_status()
                
                # Update visualization if available
                if self.visualizer:
                    self.visualizer.process_events()
                    self.visualizer.update_display()
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping passive monitor...")
            self.running = False
            
            # Stop visualization
            if self.visualizer:
                self.visualizer.stop_visualization()
                print("🎨 Visualization stopped")

def main():
    parser = argparse.ArgumentParser(description='Passive PHYSEC Protocol Monitor')
    parser.add_argument('--alice-ip', required=True, help='Alice node IP address')
    parser.add_argument('--bob-ip', required=True, help='Bob node IP address')
    parser.add_argument('--alice-port', type=int, default=8001, help='Alice port (default: 8001)')
    parser.add_argument('--bob-port', type=int, default=8002, help='Bob port (default: 8002)')
    
    args = parser.parse_args()
    
    print("👁️  Passive PHYSEC Protocol Monitor")
    print("=" * 50)
    
    monitor = PassivePHYSECMonitor(
        alice_ip=args.alice_ip,
        bob_ip=args.bob_ip,
        alice_port=args.alice_port,
        bob_port=args.bob_port
    )
    
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
