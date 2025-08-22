#!/usr/bin/env python3
"""
Network connectivity test for PHYSEC distributed setup
"""

import socket
import time
import argparse
import threading
import json

def test_server(port):
    """Test server functionality"""
    print(f"🔧 Starting test server on port {port}...")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(1)
    
    print(f"✅ Test server listening on port {port}")
    print("📡 Waiting for client connection...")
    
    try:
        client_socket, addr = server_socket.accept()
        print(f"✅ Client connected from {addr}")
        
        # Receive message
        data = client_socket.recv(1024)
        message = data.decode('utf-8').strip()
        print(f"📨 Received message: {message}")
        
        # Send response
        response = "Hello from server!"
        client_socket.send(response.encode('utf-8'))
        print(f"📤 Sent response: {response}")
        
        client_socket.close()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        server_socket.close()

def test_client(host, port):
    """Test client functionality"""
    print(f"🔧 Connecting to {host}:{port}...")
    
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(10.0)
        client_socket.connect((host, port))
        print(f"✅ Connected to {host}:{port}")
        
        # Send message
        message = "Hello from client!"
        client_socket.send(message.encode('utf-8'))
        print(f"📤 Sent message: {message}")
        
        # Receive response
        data = client_socket.recv(1024)
        response = data.decode('utf-8').strip()
        print(f"📨 Received response: {response}")
        
        client_socket.close()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Client error: {e}")

def test_ping(host):
    """Test basic connectivity"""
    import subprocess
    
    print(f"🏓 Testing ping to {host}...")
    try:
        result = subprocess.run(['ping', '-c', '3', host], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ping to {host} successful")
            # Extract timing info
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'time=' in line:
                    print(f"   📊 {line.split('time=')[1].split()[0]}")
        else:
            print(f"❌ Ping to {host} failed")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Ping test failed: {e}")

def test_physec_ports(alice_ip, bob_ip):
    """Test PHYSEC specific ports"""
    print("🔧 Testing PHYSEC ports...")
    
    # Test Alice port 8001
    print(f"\n📡 Testing Alice port (8001) on {alice_ip}:")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        result = sock.connect_ex((alice_ip, 8001))
        if result == 0:
            print("✅ Alice port 8001 is open")
        else:
            print("❌ Alice port 8001 is closed or unreachable")
        sock.close()
    except Exception as e:
        print(f"❌ Error testing Alice port: {e}")
    
    # Test Bob port 8002
    print(f"\n📡 Testing Bob port (8002) on {bob_ip}:")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        result = sock.connect_ex((bob_ip, 8002))
        if result == 0:
            print("✅ Bob port 8002 is open")
        else:
            print("❌ Bob port 8002 is closed or unreachable")
        sock.close()
    except Exception as e:
        print(f"❌ Error testing Bob port: {e}")

def main():
    parser = argparse.ArgumentParser(description='PHYSEC Network Test')
    parser.add_argument('--mode', choices=['server', 'client', 'ping', 'ports'], 
                       required=True, help='Test mode')
    parser.add_argument('--host', help='Target host (for client/ping mode)')
    parser.add_argument('--port', type=int, default=9999, help='Port number')
    parser.add_argument('--alice-ip', help='Alice IP address (for ports test)')
    parser.add_argument('--bob-ip', help='Bob IP address (for ports test)')
    
    args = parser.parse_args()
    
    print("🌐 PHYSEC Network Connectivity Test")
    print("=" * 40)
    
    if args.mode == 'server':
        test_server(args.port)
    elif args.mode == 'client':
        if not args.host:
            print("❌ --host required for client mode")
            return
        test_client(args.host, args.port)
    elif args.mode == 'ping':
        if not args.host:
            print("❌ --host required for ping mode")
            return
        test_ping(args.host)
    elif args.mode == 'ports':
        if not args.alice_ip or not args.bob_ip:
            print("❌ --alice-ip and --bob-ip required for ports mode")
            return
        test_physec_ports(args.alice_ip, args.bob_ip)

if __name__ == "__main__":
    main()
