# 🌐 PHYSEC Network Deployment Guide

Deploy PHYSEC across multiple computers with distributed visualization.

## 📋 Prerequisites

- 3 computers on the same network
- 2 PlutoSDR devices 
- GNU Radio 3.10+ and gr-PHYSEC on all computers
- Python 3.6+ with matplotlib, numpy

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Computer A    │    │   Computer B    │    │   Computer C    │
│     Alice       │◄──►│      Bob        │◄──►│  Visualization  │
│  PlutoSDR #1    │    │  PlutoSDR #2    │    │    Monitor      │
│ 192.168.1.10    │    │ 192.168.1.20    │    │ 192.168.1.30    │
│ Port: 8001      │    │ Port: 8002      │    │ Port: 9000      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Setup Instructions

### Step 1: Network Discovery

On each computer, find the network IP:
```bash
hostname -I
```

Example results:
- Computer A (Alice): `192.168.1.10`
- Computer B (Bob): `192.168.1.20`  
- Computer C (Monitor): `192.168.1.30`

### Step 2: PlutoSDR Configuration

**Computer A (Alice):**
```bash
# Test PlutoSDR connection
ssh root@192.168.2.1

# Edit config
vim /workspace/siwn/siwn-node/config.json
```
Set: `"ip": "192.168.2.1"`

**Computer B (Bob):**
```bash
# Test PlutoSDR connection  
ssh root@192.168.3.1

# Edit config
vim /workspace/siwn/siwn-node/config.json
```
Set: `"ip": "192.168.3.1"`

### Step 3: Firewall Configuration

On all computers, open required ports:
```bash
# Ubuntu/Debian
sudo ufw allow 8001
sudo ufw allow 8002
sudo ufw allow 8101
sudo ufw allow 8102

# CentOS/RHEL
sudo firewall-cmd --add-port=8001/tcp --permanent
sudo firewall-cmd --add-port=8002/tcp --permanent
sudo firewall-cmd --add-port=8101/tcp --permanent
sudo firewall-cmd --add-port=8102/tcp --permanent
sudo firewall-cmd --reload
```

## 🚀 Deployment Options

### Option 1: Manual Control Layer (Recommended)

**Computer A (Alice):**
```bash
cd /workspace/siwn/gr-PHYSEC/examples
python3 control_layer.py --node alice --peer-host 192.168.1.20
```

**Computer B (Bob):**
```bash
cd /workspace/siwn/gr-PHYSEC/examples
python3 control_layer.py --node bob --peer-host 192.168.1.10
```

**Computer C (Visualization):**
```bash
cd /workspace/siwn/gr-PHYSEC/examples
python3 network_demo.py --alice-host 192.168.1.10 --bob-host 192.168.1.20
```

### Option 2: Automated Network Demo

Run this command sequence across all computers:

**Computer A:**
```bash
# Copy repository to all computers first
scp -r /workspace/siwn user@192.168.1.20:/workspace/
scp -r /workspace/siwn user@192.168.1.30:/workspace/

# Start Alice
python3 control_layer.py --node alice --peer-host 192.168.1.20 --monitor-port 8101
```

**Computer B:**
```bash
# Start Bob
python3 control_layer.py --node bob --peer-host 192.168.1.10 --monitor-port 8102
```

**Computer C:**
```bash
# Start visualization monitor
python3 network_demo.py --alice-host 192.168.1.10 --alice-port 8001 \
                        --bob-host 192.168.1.20 --bob-port 8002
```

## 🔍 Troubleshooting

### Network Connectivity
```bash
# Test connectivity between computers
ping 192.168.1.10  # From Computer B/C to A
ping 192.168.1.20  # From Computer A/C to B  
ping 192.168.1.30  # From Computer A/B to C

# Test port connectivity
telnet 192.168.1.10 8001  # Test Alice port
telnet 192.168.1.20 8002  # Test Bob port
```

### PlutoSDR Issues
```bash
# Test PlutoSDR on each computer
ssh root@192.168.2.1  # Alice's PlutoSDR
ssh root@192.168.3.1  # Bob's PlutoSDR

# Run PlutoSDR discovery
python3 -c "
from gnuradio import iio
print('PlutoSDR URI:', iio.get_pluto_uri())
"
```

### Firewall Issues
```bash
# Temporarily disable firewall for testing
sudo ufw disable  # Ubuntu
sudo systemctl stop firewalld  # CentOS

# Test, then re-enable with proper rules
```

## 📊 Expected Behavior

**Successful Deployment:**
1. ✅ Alice connects to PlutoSDR #1 (`Connected to PlutoSDR sink/source`)
2. ✅ Bob connects to PlutoSDR #2 (`Connected to PlutoSDR sink/source`)  
3. ✅ Network communication established between Alice ↔ Bob
4. ✅ Visualization computer receives monitoring data
5. ✅ Dynamic dashboard shows real-time protocol execution
6. ✅ Both nodes complete 9-step PHYSEC protocol with hardware

**Performance:**
- **Key Generation Time:** ~7-8 seconds with real hardware
- **Success Rate:** >95% with good RF conditions
- **Network Latency:** <100ms for local network

## 🛡️ Security Considerations

- **Network Security:** Use VPN or isolated network for production
- **PlutoSDR Access:** Change default passwords on PlutoSDR devices
- **Firewall:** Only open required ports, close after testing
- **Key Management:** Secure final generated keys appropriately

## 📈 Monitoring

The visualization computer will show:
- Real-time protocol steps for both Alice and Bob
- Live IQ sample plots from both PlutoSDR devices
- Spectrogram analysis of RF channel characteristics  
- Success rate and timing statistics across multiple runs
- Bit disagreement rates and reconciliation performance

Ready for distributed quantum key generation! 🚀
