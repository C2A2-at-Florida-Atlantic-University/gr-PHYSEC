#!/usr/bin/env python3
"""
Distributed PHYSEC Demo with Multiple Runs
Shows how to run Alice and Bob on separate computers with multiple protocol executions
"""

import subprocess
import time
import argparse
import sys

def show_usage():
    """Display usage instructions"""
    print("🌐 " + "=" * 60)
    print("   DISTRIBUTED PHYSEC DEMO - MULTIPLE RUNS")
    print("=" * 64)
    print()
    print("📋 This script helps you run the distributed PHYSEC protocol")
    print("   with multiple runs across separate computers.")
    print()
    print("🚀 Usage Examples:")
    print()
    print("   1. Single run (default):")
    print("      python3 distributed_demo.py --alice-ip 192.168.0.5 --bob-ip 192.168.0.2")
    print()
    print("   2. Multiple runs:")
    print("      python3 distributed_demo.py --alice-ip 192.168.0.5 --bob-ip 192.168.0.2 --runs 5")
    print()
    print("   3. Multiple runs with custom delay:")
    print("      python3 distributed_demo.py --alice-ip 192.168.0.5 --bob-ip 192.168.0.2 --runs 3 --delay 5.0")
    print()
    print("💻 Commands to run on each computer:")
    print()
    print("   On Alice (192.168.0.5):")
    print("     python3 control_layer.py --node alice --peer-host 192.168.0.2 --runs 5")
    print()
    print("   On Bob (192.168.0.2):")
    print("     python3 control_layer.py --node bob --peer-host 192.168.0.5 --runs 5")
    print()
    print("🔍 Network Testing:")
    print("   python3 network_monitor.py --alice-ip 192.168.0.5 --bob-ip 192.168.0.2")
    print()

def test_connectivity(alice_ip, bob_ip):
    """Test basic network connectivity"""
    print("🔧 Testing network connectivity...")
    
    try:
        # Test ping to Alice
        print(f"🏓 Testing ping to Alice ({alice_ip})...")
        result = subprocess.run(['ping', '-c', '1', alice_ip], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Alice ({alice_ip}) is reachable")
        else:
            print(f"❌ Alice ({alice_ip}) is not reachable")
            return False
    except Exception as e:
        print(f"❌ Error testing Alice connectivity: {e}")
        return False
    
    try:
        # Test ping to Bob
        print(f"🏓 Testing ping to Bob ({bob_ip})...")
        result = subprocess.run(['ping', '-c', '1', bob_ip], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Bob ({bob_ip}) is reachable")
        else:
            print(f"❌ Bob ({bob_ip}) is not reachable")
            return False
    except Exception as e:
        print(f"❌ Error testing Bob connectivity: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Distributed PHYSEC Demo')
    parser.add_argument('--alice-ip', required=True, help='Alice node IP address')
    parser.add_argument('--bob-ip', required=True, help='Bob node IP address')
    parser.add_argument('--runs', type=int, default=1, help='Number of protocol runs (default: 1)')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between runs in seconds (default: 2.0)')
    parser.add_argument('--test-only', action='store_true', help='Only test connectivity, don\'t show commands')
    
    args = parser.parse_args()
    
    print("🌐 Distributed PHYSEC Demo")
    print("=" * 40)
    
    # Test connectivity
    if not test_connectivity(args.alice_ip, args.bob_ip):
        print("\n❌ Network connectivity test failed!")
        print("   Please check your network configuration and try again.")
        sys.exit(1)
    
    print("\n✅ Network connectivity test passed!")
    
    if args.test_only:
        return
    
    # Show commands
    print(f"\n🚀 Ready to run {args.runs} protocol execution(s)!")
    print("=" * 60)
    print()
    print("💻 COMMANDS TO RUN ON EACH COMPUTER:")
    print()
    print(f"📱 ALICE ({args.alice_ip}):")
    print(f"   python3 control_layer.py --node alice --peer-host {args.bob_ip} --runs {args.runs} --delay {args.delay}")
    print()
    print(f"📱 BOB ({args.bob_ip}):")
    print(f"   python3 control_layer.py --node bob --peer-host {args.alice_ip} --runs {args.runs} --delay {args.delay}")
    print()
    print("🔍 MONITORING (this computer):")
    print(f"   python3 network_monitor.py --alice-ip {args.alice_ip} --bob-ip {args.bob_ip}")
    print()
    print("📋 EXECUTION ORDER:")
    print("   1. Start Bob first (he waits for connections)")
    print("   2. Start Alice (she initiates the protocol)")
    print("   3. Watch the protocol execute {args.runs} time(s)")
    print("   4. Monitor network status from this computer")
    print()
    print("💡 TIPS:")
    print("   • Keep the network monitor running to see real-time status")
    print("   • Each run will take ~7-8 seconds with real hardware")
    print("   • Check both Jetson Orins for detailed protocol logs")
    print("   • Use Ctrl+C to stop any process")
    print()
    print("🎯 Expected Results:")
    print(f"   • {args.runs} complete PHYSEC protocol execution(s)")
    print("   • Real-time key generation on both nodes")
    print("   • Success/failure statistics for each run")
    print("   • Network connectivity monitoring")

if __name__ == "__main__":
    main()
