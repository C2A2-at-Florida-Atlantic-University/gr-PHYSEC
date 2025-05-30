# caaiSDR

This repository contains python scrypts for  sending, receiving, plotting, and collecting data using the Pluto SDR.

Requirements:
- Python >= 3.9
- Scipy
- Matplotlib
- Numpy
- Cython
- Rust
- PlutoSDR Drivers (libiio, libad9361-iio, pyadi-iio)
- flask==2.1.3
- Werkzeug==2.1.2
- flask_classful



Source - Pluto SDR Python: https://pysdr.org/content/pluto.html

Images available for devices with all dependencies installed:
- Jetson Nano: https://fau-my.sharepoint.com/:u:/g/personal/josesanchez2019_fau_edu1/EfaPn72ZDNdPq8F2DZn9kIsBDCpN8DJgT0qazN6VNALlFg?e=lQov05
- Raspberry pi 3b+: https://fau-my.sharepoint.com/:u:/g/personal/josesanchez2019_fau_edu1/EdSnLZC9MWBOo3iSYWtfOxUBaSN2CgeJ1eD4EVUskSNkXQ?e=JS3wfr

Arguments:

* Command Example: python3 main.py [mode] [options] [srd number]

* Modes:
  - tx
    - input: waveform type (synusoid, chirp, data)
  - rx
    - input: number of packages to receive

* Options:
  - none
  - plot
  - collect (only for rx mode)
    - input: dataset type (RFF, RSSI)

* SDR Number:
  - Number assigned on SDR configuration for communicating with SDR API

EXAMPLES:

Transmitting Data:
- Only transmitting: python3 main.py tx none 1
- Plotting: python3 main.py tx plot 1

Receiving Data:
- Only Receiving: python3 main.py rx none 1
- Plotting: python3 main.py rx plot 1
- Collecting: python3 main.py rx collect 1

Device Setup:

![Infrastructure-less Wireless Network-Device Setup Training drawio (1)](https://user-images.githubusercontent.com/46358777/189373699-9cf44d41-7c8d-4590-b9c8-f8c8f6239f4d.png)


