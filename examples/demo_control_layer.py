#!/usr/bin/env python3
"""
Demo script for PHYSEC Control Layer
Shows step-by-step execution of the key generation protocol
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from control_layer import Alice, Bob
import threading
import time
import logging
import numpy as np

try:
    from dynamic_visualization import start_dynamic_visualization, stop_dynamic_visualization
    DYNAMIC_VIZ_AVAILABLE = True
except ImportError:
    DYNAMIC_VIZ_AVAILABLE = False
    start_dynamic_visualization = None
    stop_dynamic_visualization = None

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_protocol(max_runs=5):
    """Demonstrate the 9-step PHYSEC protocol multiple times"""
    
    print("=" * 60)
    print("PHYSEC CONTROL LAYER DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("This demo shows the 9-step quantum key generation protocol:")
    print("1. Alice sends key generation request")
    print("2. Bob accepts and transmits sinusoidal probe")
    print("3. Alice collects samples and transmits her probe")
    print("4. Bob collects samples from Alice")
    print("5. Both process samples through PHYSEC pipeline")
    print("6. Alice generates and sends parity bits")
    print("7. Bob performs reconciliation")
    print("8. Both perform privacy amplification")
    print("9. Both exchange encrypted messages")
    print()
    print(f"Running {max_runs} iterations to collect statistics...")
    print("=" * 60)
    print()
    
    # Start dynamic visualization if available
    visualizer = None
    if DYNAMIC_VIZ_AVAILABLE:
        print("🎬 Starting dynamic visualization...")
        visualizer = start_dynamic_visualization(max_runs)
        print("✅ Dynamic visualization started")
    
    # Create nodes
    print("Creating Alice and Bob nodes...")
    alice = Alice()
    bob = Bob()
    
    print(f"✅ Alice: Listen on port {alice.listen_port}, Connect to {alice.peer_host}:{alice.peer_port}")
    print(f"✅ Bob: Listen on port {bob.listen_port}, Connect to {bob.peer_host}:{bob.peer_port}")
    print()
    
    # Start Bob's server first
    print("Starting Bob's server...")
    bob_thread = threading.Thread(target=bob.start_server)
    bob_thread.daemon = True
    bob_thread.start()
    
    # Wait for Bob to start
    # time.sleep(1)
    print("✅ Bob's server started")
    
    # Start Alice's server
    print("Starting Alice's server...")
    alice_thread = threading.Thread(target=alice.start_server)
    alice_thread.daemon = True
    alice_thread.start()
    
    # Wait for Alice to start
    time.sleep(1)
    print("✅ Alice's server started")
    print()
    
    # Begin the protocol
    print("🚀 Starting PHYSEC Key Generation Protocol...")
    print()
    
    try:
        # Run multiple iterations
        completed_runs = 0
        
        for run in range(1, max_runs + 1):
            print(f"\n{'='*20} RUN #{run} {'='*20}")
            
            # Reset nodes for new run
            alice.reset_for_new_run()
            bob.reset_for_new_run()
            
            # Reset visualization for new run
            if visualizer:
                visualizer.reset_for_new_run()
            
            # Alice initiates the protocol
            start_time = time.time()
            alice.start_key_generation()
            
            # Monitor this run
            print(f"⏳ Running protocol iteration {run}...")
            
            for i in range(240):  # Increased timeout to 240 seconds
                # time.sleep(1)
                
                # Update visualization display in main thread
                if visualizer:
                    visualizer.process_events()
                
                # Check states
                alice_state = getattr(alice, 'state', 'unknown')
                bob_state = getattr(bob, 'state', 'unknown')
                
                if i % 10 == 0:  # Update every 10 seconds for faster runs
                    print(f"   Status - Alice: {alice_state}, Bob: {bob_state}")
                    # Force display update every 10 seconds
                    if visualizer:
                        visualizer.update_display()
                
                # Check if both nodes have completed
                if alice_state == 'key_ready' and bob_state == 'key_ready':
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                    print(f"✅ Run #{run} completed successfully!")
                    completed_runs += 1
                    
                    # Update statistics
                    alice.update_statistics(bob)
                    
                    # Update dynamic visualization with final statistics
                    if visualizer:
                        # Calculate BDR for this run
                        if alice.quantized_bits and bob.quantized_bits:
                            alice_bits = np.frombuffer(alice.quantized_bits, dtype=np.uint8)
                            bob_bits = np.frombuffer(bob.quantized_bits, dtype=np.uint8)
                            disagreements = np.sum(alice_bits != bob_bits)
                            bdr = disagreements / len(alice_bits)
                            visualizer.add_run_statistics(bdr, True, duration_ms)
                            print(f"📊 Run #{run} statistics updated in visualization (BDR: {bdr:.4f}, Time: {duration_ms:.0f}ms)")
                    
                    # Static plots are now replaced by dynamic visualization
                    print(f"📊 Dynamic visualization updated for run #{run}")
                    
                    break
                
                # Check if reconciliation failed
                elif alice_state == 'reconciliation_failed' or bob_state == 'reconciliation_failed':
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                    print(f"❌ Run #{run} failed due to reconciliation error")
                    # Update failure statistics
                    alice.reconciliation_success_history.append(False)
                    bob.reconciliation_success_history.append(False)
                    if alice.quantized_bits is not None and bob.quantized_bits is not None:
                        alice.update_statistics(bob)
                        # Update visualization with failure statistics
                        if visualizer:
                            alice_bits = np.frombuffer(alice.quantized_bits, dtype=np.uint8)
                            bob_bits = np.frombuffer(bob.quantized_bits, dtype=np.uint8)
                            disagreements = np.sum(alice_bits != bob_bits)
                            bdr = disagreements / len(alice_bits)
                            visualizer.add_run_statistics(bdr, False, duration_ms)
                            print(f"📊 Run #{run} failure statistics updated in visualization (Time: {duration_ms:.0f}ms)")
                    break
                    
            else:
                print(f"⏰ Run #{run} timeout reached")
                # Proper cleanup on timeout to release resources
                try:
                    print("🧹 Cleaning up resources due to timeout...")
                    alice.cleanup_active_flowgraphs()
                    bob.cleanup_active_flowgraphs()
                except Exception as e:
                    print(f"⚠️  Cleanup warning during timeout: {e}")
                alice.reconciliation_success_history.append(False)
                bob.reconciliation_success_history.append(False)
            
            # Short pause between runs
            if run < max_runs:
                print("Preparing for next run...")
                # time.sleep(2)
        
        print(f"\n🎊 DEMO COMPLETE! Finished {completed_runs}/{max_runs} successful runs")
        
        if completed_runs > 0:
            print(f"\n📈 Final Statistics:")
            print(f"   Success Rate: {completed_runs/max_runs:.1%}")
            if alice.bit_disagreement_history:
                avg_ber = np.mean(alice.bit_disagreement_history)
                std_ber = np.std(alice.bit_disagreement_history) if len(alice.bit_disagreement_history) > 1 else 0
                print(f"   Average BER: {avg_ber:.4f} ± {std_ber:.4f}")
            if alice.correlation_history:
                avg_corr = np.mean(alice.correlation_history)
                print(f"   Average IQ Correlation: {avg_corr:.4f}")
        
    except KeyboardInterrupt:
        print()
        print("⏹️  Demo interrupted by user")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    finally:
        print()
        print("🛑 Cleaning up...")
        try:
            alice.cleanup()
            bob.cleanup()
            # Stop dynamic visualization
            if DYNAMIC_VIZ_AVAILABLE and visualizer:
                print("🛑 Stopping dynamic visualization...")
                stop_dynamic_visualization()
            # Give time for sockets to fully close
            # time.sleep(0.5)
            print("✅ Demo completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
        print()
        print("=" * 60)
        print("PHYSEC CONTROL LAYER DEMO FINISHED")
        print("=" * 60)

def explain_protocol():
    """Explain the protocol in detail"""
    
    print("PHYSEC 9-Step Protocol Overview")
    print("=" * 40)
    print("1. Alice sends key generation request")
    print("2. Bob accepts and transmits sinusoidal probe")
    print("3. Alice collects samples and transmits her probe")
    print("4. Bob collects samples from Alice")
    print("5. Both process samples through PHYSEC pipeline")
    print("6. Alice generates and sends parity bits")
    print("7. Bob performs reconciliation")
    print("8. Both perform privacy amplification")
    print("9. Both exchange encrypted messages")
    print()
    print("Security Properties:")
    print("- Information-theoretic security")
    print("- Forward secrecy")
    print("- Error correction via reconciliation")
    print("- Privacy amplification")
    print()

def cleanup_system_resources():
    """Clean up any lingering system resources"""
    import subprocess
    import signal
    import os
    
    print("🧹 Cleaning up system resources...")
    
    try:
        # Kill any lingering GNU Radio processes (but not the current process)
        current_pid = os.getpid()
        try:
            # Get list of processes to avoid killing ourselves
            result = subprocess.run(['pgrep', '-f', 'python.*demo_control_layer'], 
                                  capture_output=True, text=True, check=False)
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid and int(pid) != current_pid:
                        subprocess.run(['kill', pid], check=False,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
            
        # Clean up any GNU Radio related processes
        subprocess.run(['pkill', '-f', 'gr-'], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Remove any temp files that might hold device handles
        subprocess.run(['rm', '-f', '/tmp/sinusoidal_probe_output.dat'], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['rm', '-f', '/tmp/iq_samples_output.dat'], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Give the system a moment to release resources
        time.sleep(1.0)
        print("✅ System resources cleaned up")
        
    except Exception as e:
        print(f"⚠️  Could not clean all resources: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PHYSEC Control Layer Demo')
    parser.add_argument('--explain', action='store_true', 
                       help='Show detailed protocol explanation')
    parser.add_argument('--run', action='store_true',
                       help='Run the interactive demo')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of protocol runs for statistics (default: 3)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up system resources and exit')
    
    args = parser.parse_args()
    
    if args.explain:
        explain_protocol()
    elif args.cleanup:
        cleanup_system_resources()
    elif args.run:
        demo_protocol(max_runs=args.runs)
    else:
        print("PHYSEC Control Layer Demo")
        print("Usage:")
        print("  python3 demo_control_layer.py --explain            # Show protocol explanation")
        print("  python3 demo_control_layer.py --run                # Run interactive demo (3 runs)")
        print("  python3 demo_control_layer.py --run --runs 5       # Run demo with 5 iterations")
        print("  python3 demo_control_layer.py --cleanup            # Clean up lingering resources")
        print()
        print("For the full protocol with hardware:")
        print("  python3 control_layer.py --node alice")
        print("  python3 control_layer.py --node bob")
