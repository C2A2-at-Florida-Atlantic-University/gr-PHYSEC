# This module contains the Time Division Demultiplexer (TDD) class for bidirectional communication.


import time


class TDD():
    """
    This module contains the Time Division Demultiplexer (TDD) class for bidirectional communication.
    The TDD class is responsible for managing the time slots for data transmission and reception.
    It allows for the setting of input data and retrieval of output data.
    It uses the BPSK (Binary Phase Shift Keying) modulation scheme for data transmission.
    It reads data from a socket and sends it to the receiver.
    """

    def __init__(self, name, TX, RX, time_frequency=1):
        self.name = name
        self.TX = TX
        self.RX = RX
        self.input = None
        self.output = None
        self.time_frequency = time_frequency # Time frequency for TDD in seconds
        self.starting_time = None

    def pair(self):
        """
        Sends data to the receiver for pairing and figuring out time slots
        """
        print(f"{self.name} pairing with receiver...")
        # Simulate pairing process
        self.output = self.input
        print(f"{self.name} paired with receiver. Output set to input.")
        return self.output
    
    def set_input(self, data):
        """
        Sets the input data for transmission.
        """
        self.input = data
        print(f"{self.name} input set to: {self.input}")
    
    def read_socket(self, socket):
        """
        Reads data from the socket and sets it as input data.
        """
        self.input = socket.recv(1024)
        print(f"{self.name} received data: {self.input}")
        return self.input
    
    def transmit_data(self, socket):
        """
        Transmits the input data using the modulator.
        """
        if self.input is not None:
            # Send data over the modulator
            self.TX.transmit(self.input, self.time_frequency, self.starting_time)
        else:
            print(f"{self.name} has no data to transmit.")
            
    def receive_data(self):
        """
        Receives data from the socket and sets it as output data.
        """
        self.output = self.RX.receive()
            
    def toggle_TX_RX(self):
        """
        Toggles between TX and RX modes.
        """
        # Check if the current time slot is for TX or RX using starting time and current time
        current_time = time.time()
        time_passed = (current_time - self.starting_time)
        if time_passed % (2 * self.time_frequency) < self.time_frequency:
            self.RX.toggle()
            print(f"{self.name} toggled to RX mode.")
        else:
            self.TX.toggle()
            print(f"{self.name} toggled to TX mode.")
        
        