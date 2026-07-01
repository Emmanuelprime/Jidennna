"""
Layer 5: Hardware Communication Layer
"""
import serial
import struct
import time
import threading
from collections import deque
import numpy as np

class CommunicationInterface:
    """
    Communication interface between Jetson Nano and ESP32
    
    Protocol:
    - Control Packet: [0xAA, seq_num, rpm_left(2), rpm_right(2), checksum]
    - Feedback Packet: [0xBB, seq_num, rpm_left(2), rpm_right(2), 
                       theta(2), x(4), y(4), checksum]
    - Feedback Packet with distance sensors:
      [0xBB, seq_num, rpm_left(2), rpm_right(2), theta(2), x(4), y(4),
       distance_left_mm(2), distance_center_mm(2), distance_right_mm(2), checksum]
    """
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        
        # Threading
        self.rx_thread = None
        self.running = False
        
        # Data buffers
        self.latest_feedback = {}
        self.feedback_queue = deque(maxlen=100)
        self.command_queue = deque(maxlen=10)
        
        # Sequence numbers
        self.tx_seq = 0
        self.rx_seq = 0
        
        # Statistics
        self.packets_sent = 0
        self.packets_received = 0
        self.comm_errors = 0
        
    def connect(self):
        """Establish connection with ESP32"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=0.1
            )
            self.is_connected = True
            self.running = True
            
            # Start receive thread
            self.rx_thread = threading.Thread(target=self._receive_loop)
            self.rx_thread.daemon = True
            self.rx_thread.start()
            
            print(f"Connected to ESP32 on {self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        self.running = False
        if self.rx_thread:
            self.rx_thread.join(timeout=1.0)
        
        if self.serial_conn:
            self.serial_conn.close()
        
        self.is_connected = False
        print("Disconnected from ESP32")
    
    def send_control(self, rpm_left, rpm_right):
        """
        Send control command to ESP32
        
        Args:
            rpm_left: left wheel RPM
            rpm_right: right wheel RPM
        """
        if not self.is_connected:
            print("Not connected")
            return False
        
        try:
            # Build packet
            packet = bytearray()
            packet.append(0xAA)
            packet.append(self.tx_seq & 0xFF)
            
            # RPM values
            rpm_left_int = int(rpm_left * 100)
            rpm_right_int = int(rpm_right * 100)
            
            packet.extend(rpm_left_int.to_bytes(2, byteorder='little', signed=True))
            packet.extend(rpm_right_int.to_bytes(2, byteorder='little', signed=True))
            
            # Checksum
            checksum = sum(packet[1:]) & 0xFF
            packet.append(checksum)
            
            # Send packet
            self.serial_conn.write(bytes(packet))
            self.packets_sent += 1
            self.tx_seq = (self.tx_seq + 1) & 0xFF
            
            return True
            
        except Exception as e:
            print(f"Send error: {e}")
            self.comm_errors += 1
            return False
    
    def _receive_loop(self):
        """Background thread for receiving feedback"""
        buffer = bytearray()
        
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    # Read available data
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    buffer.extend(data)
                    
                    # Process complete packets
                    while len(buffer) >= 17:  # Legacy packet size
                        # Look for start byte
                        if buffer[0] != 0xBB:
                            buffer.pop(0)
                            continue
                        
                        packet_len = self._detect_feedback_packet_length(buffer)
                        if packet_len is None:
                            if len(buffer) < 23:
                                break
                            buffer.pop(0)
                            self.comm_errors += 1
                            continue

                        packet = buffer[:packet_len]
                        
                        # Parse data
                        seq_num = packet[1]
                        rpm_left = int.from_bytes(packet[2:4], 'little', signed=True) / 100.0
                        rpm_right = int.from_bytes(packet[4:6], 'little', signed=True) / 100.0
                        theta = int.from_bytes(packet[6:8], 'little', signed=True) / 100.0
                        x = int.from_bytes(packet[8:12], 'little', signed=True) / 1000.0
                        y = int.from_bytes(packet[12:16], 'little', signed=True) / 1000.0
                        distances = self._parse_distance_sensors(packet)
                        
                        # Store feedback
                        feedback = {
                            'timestamp': time.time(),
                            'seq_num': seq_num,
                            'rpm_left': rpm_left,
                            'rpm_right': rpm_right,
                            'theta': theta,
                            'x': x,
                            'y': y,
                            **distances
                        }
                        
                        self.latest_feedback = feedback
                        self.feedback_queue.append(feedback)
                        self.packets_received += 1
                        
                        # Remove processed packet from buffer
                        buffer = buffer[packet_len:]
                
                else:
                    time.sleep(0.001)  # Yield CPU
                    
            except Exception as e:
                print(f"Receive error: {e}")
                self.comm_errors += 1
                time.sleep(0.01)
    
    def get_latest_feedback(self):
        """Get most recent feedback data"""
        return self.latest_feedback

    def _detect_feedback_packet_length(self, buffer):
        if len(buffer) >= 23:
            checksum = sum(buffer[1:22]) & 0xFF
            if checksum == buffer[22]:
                return 23

        if len(buffer) >= 17:
            checksum = sum(buffer[1:16]) & 0xFF
            if checksum == buffer[16]:
                return 17

        return None

    def _parse_distance_sensors(self, packet):
        if len(packet) < 23:
            return {}

        return {
            'distance_left': int.from_bytes(packet[16:18], 'little', signed=False) / 1000.0,
            'distance_center': int.from_bytes(packet[18:20], 'little', signed=False) / 1000.0,
            'distance_right': int.from_bytes(packet[20:22], 'little', signed=False) / 1000.0
        }
    
    def get_comm_stats(self):
        """Get communication statistics"""
        return {
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'errors': self.comm_errors,
            'connected': self.is_connected
        }
