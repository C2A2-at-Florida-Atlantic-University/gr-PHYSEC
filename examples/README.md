# PHYSEC Control Layer

Bidirectional TCP/IP communication for quantum key generation between Alice and Bob nodes using the PHYSEC library.

## 🚀 Quick Start

### Interactive Demo (No Hardware Required)
```bash
# Run demo with 3 iterations (default)
python3 demo_control_layer.py --run

# Run demo with 5 iterations for better statistics
python3 demo_control_layer.py --run --runs 5
```

### Protocol Explanation
```bash
python3 demo_control_layer.py --explain
```

### Full Protocol with Hardware
```bash
# Terminal 1 (Bob - Listener)
python3 control_layer.py --node bob --monitor-ip 192.168.0.142

# Terminal 2 (Alice - Initiator)  
python3 control_layer.py --node alice --monitor-ip 192.168.0.142
```

### Network Deployment (Multi-Computer)
```bash
# On Bob's machine (with PlutoSDR #2)
python3 control_layer.py --node bob --peer-host alice_ip_address --monitor-ip monitor_ip_address

# On Alice's machine (with PlutoSDR #1)
python3 control_layer.py --node alice --peer-host bob_ip_address --monitor-ip monitor_ip_address

# On visualization machine
python3 protocol_monitor.py --alice-ip alice_ip --bob-ip bob_ip
```

📖 **For detailed distributed setup:** See [NETWORK_DEPLOYMENT.md](NETWORK_DEPLOYMENT.md)

## 📁 Files

### Core Files
- **`control_layer.py`** - Main PHYSEC protocol implementation with automatic data pushing
- **`demo_control_layer.py`** - Interactive demonstration with dynamic visualization
- **`dynamic_visualization.py`** - Real-time dashboard for protocol monitoring
- **`protocol_monitor.py`** - Simple push-only monitoring with real-time data visualization
- **`network_demo.py`** - Network monitoring for distributed deployment

### Modular Flowgraphs
- **`flowgraphs/`** - Modular GNU Radio implementations
  - `sinusoidal_probe.py` - RF signal generation
  - `iq_receiver.py` - RF signal collection
  - `physec_processor.py` - Spectrogram and feature extraction
  - `parity_generator.py` - Reed-Solomon error correction
  - `reconciliator.py` - Key reconciliation
  - `privacy_amplifier.py` - SHA3-512 key derivation

### Reference & Analysis
- **`flowgraphs/decoupled_physic_example.py`** - Reference GNU Radio example
- **`analyze_feature_vectors.py`** - Feature analysis utilities

## 🔧 Requirements

- GNU Radio 3.10+ with IIO support
- gr-PHYSEC library  
- PlutoSDR hardware (optional - falls back to test signals)
- Python 3.6+
- matplotlib, numpy

### PlutoSDR Configuration

The system automatically detects PlutoSDR hardware and falls back to test signals if unavailable.

**Hardware Setup:**
1. Connect PlutoSDR via USB or Ethernet
2. Verify connectivity: `ssh root@192.168.2.1` (default IP)
3. Update `/workspace/siwn/siwn-node/config.json` with correct IP if needed

**Demo Modes:**
- **Mixed Mode**: Alice uses hardware, Bob uses test signals (current default)
- **Full Hardware**: Requires two PlutoSDR devices for simultaneous operation
- **Test Mode**: Both nodes use simulated signals when no hardware detected

## 📊 Protocol Steps

1. Alice sends key generation request
2. Bob accepts and transmits sinusoidal probe  
3. Alice collects samples and transmits her probe
4. Bob collects samples from Alice
5. Both process samples through PHYSEC pipeline
6. Alice generates and sends parity bits
7. Bob performs reconciliation
8. Both perform privacy amplification
9. Both exchange encrypted messages

## 🎯 Features

- **Bidirectional TCP/IP Communication**
- **Complete 9-Step PHYSEC Protocol** 
- **GNU Radio Integration** with PlutoSDR
- **Automatic Error Correction** and reconciliation
- **Privacy Amplification** using SHA3-512
- **Real-time Visualization** of IQ samples, spectrograms, and bit analysis
- **Quality Assessment** with channel evaluation

## 📈 Dynamic Visualization

The interactive demo features a **real-time dashboard** that displays:

### Current Run Status
- **Protocol Step Indicator** - Shows current phase (transmission, processing, reconciliation, etc.)
- **IQ Sample Plots** - Live Alice and Bob IQ data from latest run  
- **Spectrogram Analysis** - Time-frequency analysis of RF channel characteristics

### Historical Statistics  
- **Bit Disagreement Rate (BDR)** - Running average across all runs
- **Success Rate Tracking** - Color-coded markers (green=success, red=failure) on trend line
- **Key Generation Timing** - Individual run times and running average
- **Reconciliation Success Rate** - Cumulative success percentage

### Features
- **Real-time Updates** - Dashboard refreshes as protocol executes
- **Multi-run Statistics** - Averages and trends across multiple iterations  
- **No File Saving** - All visualization in-memory for performance
- **matplotlib Integration** - Clean, professional plots with legends and annotations

## 📡 **Protocol Monitoring**

The `protocol_monitor.py` script provides **simple, push-only monitoring**:

### **Push-Only Data Collection Strategy**
- **No Polling**: Monitor only receives data that nodes push automatically
- **Real-time Updates**: Visualization updates immediately as data arrives
- **Simple Architecture**: Just one data collection server on port 9999

### **Real-time Data Flow**
- **IQ Samples**: Pushed immediately after collection (8192 samples)
- **Spectrograms**: Pushed after PHYSEC processing
- **Quantized Bits**: Pushed after feature quantization
- **Statistics**: Pushed after reconciliation (BDR, success, timing)

### **Network Architecture**
```
Alice ──push data──→ Monitor (Port 9999) ←──push data── Bob
```

### **Usage**
```bash
# Start monitor (receives only pushed data)
python3 protocol_monitor.py --alice-ip 192.168.0.5 --bob-ip 192.168.0.2

# Monitor will show:
# 📥 Real-time data pushes as they arrive
# 🎨 Immediate visualization updates
# 📊 Simple status display every 10 seconds
```

## 🔐 Security

- Information-theoretic security based on physical channel
- Forward secrecy through independent key generation
- Error detection and correction via Reed-Solomon codes
- Privacy amplification removes leaked information

Ready for quantum key generation! 🚀
