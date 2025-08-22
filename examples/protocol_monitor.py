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
            sock.settimeout(2)  # 2 second timeout
            sock.connect((ip, port))
            
            # Send status request
            request = {"type": "status_request"}
            sock.send(json.dumps(request).encode('utf-8') + b'\n')
            
            # Get response
            data = sock.recv(1024)
            sock.close()
            
            if data:
                response = json.loads(data.decode('utf-8').strip())
                return response
            else:
                return {"state": "no_response"}
                
        except Exception as e:
            return {"state": "connection_error", "error": str(e)}

    def request_data_from_node(self, ip, port, node_name, data_type):
        """Request specific data from a node"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # Longer timeout for data requests
            sock.connect((ip, port))
            
            # Send data request
            request = {"type": f"{data_type}_request"}
            sock.send(json.dumps(request).encode('utf-8') + b'\n')
            
            # Get response
            data = sock.recv(8192)  # Larger buffer for data
            sock.close()
            
            if data:
                response = json.loads(data.decode('utf-8').strip())
                return response
            else:
                return {"error": "no_data_response"}
                
        except Exception as e:
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
                    print(f"📊 Updated {node_name} IQ data visualization")
                
                # Update spectrogram if available
                if spectrogram_data is not None:
                    self.visualizer.update_spectrogram(node_name, spectrogram_data)
                    print(f"📊 Updated {node_name} spectrogram visualization")
                
                # Force visualization update
                self.visualizer.process_events()
                self.visualizer.update_display()
                
            except Exception as e:
                print(f"⚠️  Data visualization update error: {e}")

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
            
            if "run_number" in response:
                self.current_run = response["run_number"]
            
            # Check for changes
            state_changed = self.alice_state != old_state
            step_changed = self.alice_protocol_step != old_step
            iq_changed = self.alice_iq_available != old_iq
            spec_changed = self.alice_spectrogram_available != old_spec
            
            if state_changed or step_changed:
                print(f"🔄 Alice: {old_state}→{self.alice_state}, {old_step}→{self.alice_protocol_step}")
                self.update_visualization()
            
            # Request data if newly available
            if iq_changed and self.alice_iq_available:
                print(f"📊 Alice IQ samples newly available, requesting data...")
                iq_response = self.request_data_from_node(self.alice_ip, self.alice_port, "Alice", "iq_samples")
                if "iq_samples" in iq_response:
                    # Convert string representation back to numpy array
                    iq_data = np.array(eval(iq_response["iq_samples"]), dtype=np.complex64)
                    self.update_visualization_with_data("Alice", iq_data=iq_data)
            
            if spec_changed and self.alice_spectrogram_available:
                print(f"📊 Alice spectrogram newly available, requesting data...")
                spec_response = self.request_data_from_node(self.alice_ip, self.alice_port, "Alice", "spectrogram")
                if "spectrogram_data" in spec_response:
                    # Convert string representation back to numpy array
                    spec_data = np.array(eval(spec_response["spectrogram_data"]), dtype=np.float32)
                    self.update_visualization_with_data("Alice", spectrogram_data=spec_data)
                
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
            
            if "run_number" in response:
                self.current_run = response["run_number"]
            
            # Check for changes
            state_changed = self.bob_state != old_state
            step_changed = self.bob_protocol_step != old_step
            iq_changed = self.bob_iq_available != old_iq
            spec_changed = self.bob_spectrogram_available != old_spec
            
            if state_changed or step_changed:
                print(f"🔄 Bob: {old_state}→{self.bob_state}, {old_step}→{self.bob_protocol_step}")
                self.update_visualization()
            
            # Request data if newly available
            if iq_changed and self.bob_iq_available:
                print(f"📊 Bob IQ samples newly available, requesting data...")
                iq_response = self.request_data_from_node(self.bob_ip, self.bob_port, "Bob", "iq_samples")
                if "iq_samples" in iq_response:
                    # Convert string representation back to numpy array
                    iq_data = np.array(eval(iq_response["iq_samples"]), dtype=np.complex64)
                    self.update_visualization_with_data("Bob", iq_data=iq_data)
            
            if spec_changed and self.bob_spectrogram_available:
                print(f"📊 Bob spectrogram newly available, requesting data...")
                spec_response = self.request_data_from_node(self.bob_ip, self.bob_port, "Bob", "spectrogram")
                if "spectrogram_data" in spec_response:
                    # Convert string representation back to numpy array
                    spec_data = np.array(eval(spec_response["spectrogram_data"]), dtype=np.float32)
                    self.update_visualization_with_data("Bob", spectrogram_data=spec_data)
                
        except Exception as e:
            print(f"❌ Error monitoring Bob: {e}")

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
