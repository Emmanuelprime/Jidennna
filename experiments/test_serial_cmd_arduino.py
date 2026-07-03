#!/usr/bin/env python3
"""
Simple Serial Monitor for Robot Testing
----------------------------------------
Sends commands and displays telemetry from the robot.
"""

import serial
import time
import sys
import threading
import argparse

class SerialMonitor:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.connected = False
        
    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            self.connected = True
            self.running = True
            print(f"✅ Connected to {self.port} @ {self.baudrate} baud")
            print("="*60)
            print("Robot Commands:")
            print("  V<v>,<w>  - Set velocity (e.g., V0.3,0.5)")
            print("  F<speed>  - Forward (e.g., F0.3)")
            print("  R<speed>  - Reverse (e.g., R0.3)")
            print("  L<omega>  - Spin left (e.g., L1.0)")
            print("  B<omega>  - Spin right (e.g., B1.0)")
            print("  s         - Stop")
            print("  z         - Zero encoders")
            print("  C         - Calibrate IMU")
            print("  PING      - Check connection")
            print("  h         - Show this help")
            print("="*60)
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("🔌 Disconnected")
    
    def read_serial(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # Parse CNT messages for pretty display
                        if line.startswith('CNT,'):
                            parts = line.split(',')
                            if len(parts) >= 12:
                                try:
                                    # CNT,time,vL,vR,linear,omega,actualOmega,yaw,x,y,leftPWM,rightPWM
                                    print(f"\r📍 Pos: ({float(parts[8]):.2f}, {float(parts[9]):.2f})m  "
                                          f"Yaw: {float(parts[7]):.1f}°  "
                                          f"v: {float(parts[4]):.2f}m/s  "
                                          f"ω: {float(parts[5]):.2f}rad/s  "
                                          f"PWM: {int(parts[10])}/{int(parts[11])}", end='')
                                    sys.stdout.flush()
                                    continue
                                except:
                                    pass
                        # For other messages, print them
                        print(f"\n{line}")
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Read error: {e}")
                break
    
    def send_command(self, cmd):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{cmd}\n".encode())
                self.ser.flush()
                print(f"\n📤 Sent: {cmd}")
                return True
            except Exception as e:
                print(f"❌ Send error: {e}")
        return False
    
    def run(self):
        if not self.connect():
            return
        
        # Start reading thread
        read_thread = threading.Thread(target=self.read_serial, daemon=True)
        read_thread.start()
        
        try:
            while self.running:
                cmd = input("> ").strip()
                if not cmd:
                    continue
                
                if cmd.lower() == 'q' or cmd.lower() == 'exit':
                    break
                elif cmd.lower() == 'h':
                    print("\nCommands:")
                    print("  V<v>,<w>  - Set velocity (e.g., V0.3,0.5)")
                    print("  F<speed>  - Forward (e.g., F0.3)")
                    print("  R<speed>  - Reverse (e.g., R0.3)")
                    print("  L<omega>  - Spin left (e.g., L1.0)")
                    print("  B<omega>  - Spin right (e.g., B1.0)")
                    print("  s         - Stop")
                    print("  z         - Zero encoders")
                    print("  C         - Calibrate IMU")
                    print("  PING      - Check connection")
                    print("  q         - Quit")
                    print("  h         - Show this help")
                else:
                    self.send_command(cmd)
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        finally:
            self.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Serial Monitor for Robot Testing")
    parser.add_argument('--port', '-p', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', '-b', type=int, default=115200, help='Baud rate')
    args = parser.parse_args()
    
    monitor = SerialMonitor(args.port, args.baud)
    monitor.run()

if __name__ == "__main__":
    main()