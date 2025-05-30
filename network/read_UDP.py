import socket
import struct
import json

UDP_IP = "127.0.0.1"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
data_type = "Data"

if data_type == "IQ":
    UDP_PORT = 40860 
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)
    dataSize = 8192*8
    IQ_data = []
    while len(IQ_data)<8192:
        try:
            data, addr = sock.recvfrom(dataSize) 
            print("data complete")
            for i in range(0, len(data), 8):
                real, imag = struct.unpack('ff', data[i:i+8])
                complex_num = complex(real, imag)
                IQ_data.append(complex_num)
        except socket.timeout:
            pass

    print(IQ_data)
    reals = [z.real for z in IQ_data]
    imags = [z.imag for z in IQ_data]
else:
    UDP_PORT = 40868 
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)
    dataSize = 8192
    flag = True
    while flag:
        try:
            data, addr = sock.recvfrom(dataSize) 
            try: 
                data = json.loads(data)

                UUID = data['UUID']
                print('UUID: ', UUID)

                ts = data['ts']
                print('ts: ', ts)

                size = data['size']
                print('size: ', size)
                
                type = data['type']
                print('type: ', type)

                contents = data['data']
                print('data: ', contents)

                flag = False
            except:  # noqa: E722
                print("data retrieved not in JSON format")
                print(data)
        except socket.timeout:
            print("timeout")
            pass

