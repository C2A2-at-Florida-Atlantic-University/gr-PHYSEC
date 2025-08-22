#!/usr/bin/env python3
"""
Simple server test to debug Alice's network binding
"""

import socket
import time
import threading

def simple_server(port=8001):
    """Simple test server"""
    print(f"🔧 Starting simple test server on port {port}...")
    
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # Bind to all interfaces
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(1)
        
        print(f"✅ Server bound to 0.0.0.0:{port}")
        print(f"📡 Server listening on port {port}")
        
        # Show what we're bound to
        import subprocess
        try:
            result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if str(port) in line:
                    print(f"🔍 Netstat shows: {line.strip()}")
        except:
            pass
        
        # Wait for connections
        while True:
            print("⏳ Waiting for client connection...")
            client_socket, addr = server_socket.accept()
            print(f"✅ Client connected from {addr}")
            
            # Send welcome message
            welcome = f"Hello from Alice server on port {port}!"
            client_socket.send(welcome.encode('utf-8'))
            print(f"📤 Sent: {welcome}")
            
            # Close connection
            client_socket.close()
            print("🔌 Connection closed")
            
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server_socket.close()

if __name__ == "__main__":
    simple_server(8001)
