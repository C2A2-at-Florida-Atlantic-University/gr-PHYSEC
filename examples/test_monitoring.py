#!/usr/bin/env python3
"""
Test script to verify monitoring ports work
"""

import socket
import json
import time

def test_monitoring_port(ip, port, node_name):
    """Test connection to monitoring port"""
    print(f"🔧 Testing {node_name} monitoring port {ip}:{port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ip, port))
        
        # Send status request
        request = {
            "type": "status_request",
            "timestamp": time.time()
        }
        sock.send(json.dumps(request).encode('utf-8') + b'\n')
        
        # Wait for response
        data = sock.recv(1024)
        sock.close()
        
        if data:
            try:
                response = json.loads(data.decode('utf-8').strip())
                print(f"✅ {node_name} monitoring port working!")
                print(f"   Response: {response}")
                return True
            except json.JSONDecodeError:
                print(f"❌ {node_name} invalid JSON response")
                return False
        else:
            print(f"❌ {node_name} no response")
            return False
            
    except Exception as e:
        print(f"❌ {node_name} monitoring port failed: {e}")
        return False

def main():
    print("🔧 PHYSEC Monitoring Port Test")
    print("=" * 40)
    
    # Test monitoring ports
    alice_ip = "192.168.0.5"
    bob_ip = "192.168.0.2"
    
    print(f"\n📡 Testing monitoring ports...")
    print(f"   Alice: {alice_ip}:9001 (8001 + 1000)")
    print(f"   Bob:   {bob_ip}:9002 (8002 + 1000)")
    
    alice_ok = test_monitoring_port(alice_ip, 9001, "Alice")
    bob_ok = test_monitoring_port(bob_ip, 9002, "Bob")
    
    print(f"\n📊 Test Results:")
    print(f"   Alice monitoring: {'✅ Working' if alice_ok else '❌ Failed'}")
    print(f"   Bob monitoring:   {'✅ Working' if bob_ok else '❌ Failed'}")
    
    if alice_ok and bob_ok:
        print(f"\n🎉 All monitoring ports working!")
        print(f"   You can now use the protocol monitor:")
        print(f"   python3 protocol_monitor.py --alice-ip {alice_ip} --bob-ip {bob_ip}")
    else:
        print(f"\n⚠️  Some monitoring ports failed!")
        print(f"   Check that Alice and Bob are running with the updated control_layer.py")

if __name__ == "__main__":
    main()
