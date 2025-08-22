#!/usr/bin/env python3
"""
Network PHYSEC Demo Instructions
Provides guidance for running distributed Alice/Bob setup
"""

import sys
import os
import argparse
import time

def show_network_setup():
    """Display network setup instructions"""
    print("🌐 " + "=" * 60)
    print("   PHYSEC DISTRIBUTED NETWORK SETUP")
    print("=" * 64)
    print()
    
    print("📋 Prerequisites:")
    print("   • 3 computers on the same network")
    print("   • 2 PlutoSDR devices")
    print("   • GNU Radio 3.10+ with gr-PHYSEC on all computers")
    print()
    
    print("🏗️  Architecture:")
    print("   Computer A (Alice) ←→ Computer B (Bob) ←→ Computer C (Monitor)")
    print("   PlutoSDR #1          PlutoSDR #2          Visualization")
    print()
    
    print("🚀 Step-by-Step Commands:")
    print()
    
    # Get network details from user
    alice_ip = input("   Enter Alice computer IP (e.g., 192.168.1.10): ").strip()
    bob_ip = input("   Enter Bob computer IP (e.g., 192.168.1.20): ").strip()
    
    if not alice_ip:
        alice_ip = "192.168.1.10"
    if not bob_ip:
        bob_ip = "192.168.1.20"
    
    print()
    print("💻 Computer A (Alice):")
    print("   1. Connect PlutoSDR #1 via USB")
    print("   2. Test PlutoSDR: ssh root@192.168.2.1")
    print(f"   3. Run: python3 control_layer.py --node alice --peer-host {bob_ip}")
    print()
    
    print("💻 Computer B (Bob):")
    print("   1. Connect PlutoSDR #2 via USB")
    print("   2. Test PlutoSDR: ssh root@192.168.3.1")
    print(f"   3. Run: python3 control_layer.py --node bob --peer-host {alice_ip}")
    print()
    
    print("💻 Computer C (Visualization - Optional):")
    print("   1. For individual demos, run the demo on either Alice or Bob's computer:")
    print(f"      python3 demo_control_layer.py --run --runs 5")
    print("   2. For monitoring, SSH into Alice or Bob and watch the console output")
    print()
    
    print("🔍 Troubleshooting:")
    print("   • Test connectivity: ping <target_ip>")
    print("   • Check ports: telnet <target_ip> 8001 (Alice) or 8002 (Bob)")
    print("   • Verify PlutoSDR: Check 'Connected to PlutoSDR sink/source' messages")
    print("   • View logs: Both nodes show detailed protocol execution")
    print()
    
    print("✅ Expected Results:")
    print("   • Alice: 'Connected to PlutoSDR sink/source'")
    print("   • Bob: 'Connected to PlutoSDR sink/source'")
    print("   • Both: Complete 9-step PHYSEC protocol")
    print("   • ~7-8 second key generation with real hardware")
    print()

def show_single_computer_demo():
    """Show how to run demo on single computer with test signals"""
    print("🖥️  " + "=" * 60)
    print("   SINGLE COMPUTER DEMO (Test Signals)")
    print("=" * 64)
    print()
    
    print("For development and testing without multiple PlutoSDRs:")
    print()
    print("💻 Single Computer Setup:")
    print("   1. Alice uses PlutoSDR hardware (if available)")
    print("   2. Bob uses test signals automatically")
    print("   3. Protocol completes successfully with mixed mode")
    print()
    
    print("🚀 Run the demo:")
    print("   python3 demo_control_layer.py --run --runs 3")
    print()
    
    print("📊 Features:")
    print("   • Real-time dynamic visualization")
    print("   • Multiple run statistics")
    print("   • Success rate tracking")
    print("   • Key generation timing")
    print("   • Automatic hardware detection with fallback")
    print()

def monitor_mode():
    """Simple monitoring mode with manual setup"""
    print("📊 " + "=" * 60)
    print("   PHYSEC NETWORK MONITORING")
    print("=" * 64)
    print()
    
    print("Currently, the best way to monitor distributed PHYSEC is:")
    print()
    print("1. 📺 Console Monitoring:")
    print("   • SSH into Alice or Bob computers")
    print("   • Watch detailed protocol logs in real-time")
    print("   • See timing, success rates, and error messages")
    print()
    
    print("2. 📊 Visualization on Each Node:")
    print("   • Run demo_control_layer.py on Alice's computer")
    print("   • Run demo_control_layer.py on Bob's computer")
    print("   • Each shows their perspective of the protocol")
    print()
    
    print("3. 🔧 Future Enhancement:")
    print("   • Network monitoring interface can be added")
    print("   • Centralized dashboard for multi-node visualization")
    print("   • Real-time protocol state synchronization")
    print()
    
    print("⏳ Keeping monitor alive... Press Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")

def main():
    parser = argparse.ArgumentParser(description='PHYSEC Network Setup Helper')
    parser.add_argument('--setup', action='store_true',
                       help='Show network setup instructions')
    parser.add_argument('--single', action='store_true',
                       help='Show single computer demo instructions')
    parser.add_argument('--monitor', action='store_true',
                       help='Run monitoring mode (instructions)')
    parser.add_argument('--alice-host', help='Alice computer IP (for compatibility)')
    parser.add_argument('--bob-host', help='Bob computer IP (for compatibility)')
    
    args = parser.parse_args()
    
    if args.setup:
        show_network_setup()
    elif args.single:
        show_single_computer_demo()  
    elif args.monitor or args.alice_host or args.bob_host:
        monitor_mode()
    else:
        print("PHYSEC Network Demo")
        print("Choose an option:")
        print("  --setup     Interactive network setup guide")
        print("  --single    Single computer demo instructions")
        print("  --monitor   Monitoring instructions")
        print()
        print("Examples:")
        print("  python3 network_demo.py --setup")
        print("  python3 network_demo.py --single")
        print("  python3 network_demo.py --monitor")

if __name__ == "__main__":
    main()
