import serial
import time
import csv
import argparse
import sys
import numpy as np
from datetime import datetime

class MotorTester:
    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        
    def connect(self):
        try:
            print(f"Opening serial port {self.port}...")
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print("Port opened, waiting for Arduino to reset...")
            time.sleep(2)
            
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            print("Sending PING...")
            for attempt in range(5):
                self.ser.write(b"PING\n")
                self.ser.flush()
                time.sleep(0.5)
                
                while self.ser.in_waiting:
                    response = self.ser.readline().decode().strip()
                    if response:
                        print(f"Response: {response}")
                        if response == "READY":
                            print("Connected successfully!")
                            return True
                
                print(f"Attempt {attempt + 1} failed, retrying...")
            
            return False
            
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            return False
    
    def send_command(self, cmd):
        self.ser.reset_input_buffer()
        self.ser.write(f"{cmd}\n".encode())
        self.ser.flush()
        time.sleep(0.2)
    
    def wait_for_response(self, expected_prefix, timeout=2.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ser.in_waiting:
                line = self.ser.readline().decode().strip()
                if line.startswith(expected_prefix):
                    return line
        return None
    
    def collect_data(self, duration=5):
        data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if self.ser.in_waiting:
                line = self.ser.readline().decode().strip()
                if line.startswith("DATA,"):
                    parts = line.split(',')
                    if len(parts) == 10:
                        data.append({
                            'timestamp': int(parts[1]),
                            'left_count': int(parts[2]),
                            'right_count': int(parts[3]),
                            'left_speed': float(parts[4]),
                            'right_speed': float(parts[5]),
                            'left_pwm': int(parts[6]),
                            'right_pwm': int(parts[7]),
                            'left_dir': int(parts[8]),
                            'right_dir': int(parts[9])
                        })
            else:
                time.sleep(0.01)
        return data
    
    def test_pwm(self, left_pwm, right_pwm, duration=5):
        # First ensure motors are stopped and stream is off
        self.send_command("s")
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        
        # Start streaming
        cmd = f"STREAM:{left_pwm},{right_pwm}"
        self.send_command(cmd)
        
        response = self.wait_for_response("STREAMING:", timeout=2.0)
        if response is None:
            print(f"  Warning: No STREAMING confirmation")
        else:
            print(f"  {response}")
        
        # Collect data
        data = self.collect_data(duration)
        
        # Stop streaming
        self.send_command("STOP_STREAM")
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        
        # Stop motors
        self.send_command("s")
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        
        return data
    
    def disconnect(self):
        if self.ser:
            self.send_command("s")
            time.sleep(0.2)
            self.ser.close()

def predict_speeds(left_pwm, right_pwm):
    left_vel = 0.0373 * left_pwm - 0.3018 if left_pwm > 0 else 0
    right_vel = 0.0110 * right_pwm + 0.0350 if right_pwm > 0 else 0
    return left_vel, right_vel

def calculate_needed_pwm(target_left_vel, target_right_vel):
    left_pwm = (target_left_vel + 0.3018) / 0.0373
    right_pwm = (target_right_vel - 0.0350) / 0.0110
    return int(round(max(0, min(60, left_pwm)))), int(round(max(0, min(60, right_pwm))))

def test_model_accuracy(tester):
    print("\n" + "="*60)
    print("TEST 1: MODEL ACCURACY - FIXED PWM VALUES")
    print("="*60)
    
    pwm_pairs = [(15, 15), (25, 25), (35, 35), (45, 45), (55, 55)]
    
    for left_pwm, right_pwm in pwm_pairs:
        print(f"\n--- Testing PWM Left={left_pwm}, Right={right_pwm} ---")
        data = tester.test_pwm(left_pwm, right_pwm, duration=3)
        
        if data:
            left_speeds = [abs(d['left_speed']) for d in data if abs(d['left_speed']) > 0.001]
            right_speeds = [abs(d['right_speed']) for d in data if abs(d['right_speed']) > 0.001]
            
            if left_speeds:
                left_mean = np.mean(left_speeds)
                left_std = np.std(left_speeds)
            else:
                left_mean = 0
                left_std = 0
                
            if right_speeds:
                right_mean = np.mean(right_speeds)
                right_std = np.std(right_speeds)
            else:
                right_mean = 0
                right_std = 0
            
            pred_left, pred_right = predict_speeds(left_pwm, right_pwm)
            
            left_error = abs(left_mean - pred_left) / max(abs(pred_left), 0.001) * 100 if abs(pred_left) > 0.001 else 0
            right_error = abs(right_mean - pred_right) / max(abs(pred_right), 0.001) * 100 if abs(pred_right) > 0.001 else 0
            
            print(f"  Left  - Pred: {pred_left:.4f}, Actual: {left_mean:.4f} ± {left_std:.4f} m/s, Error: {left_error:.1f}%")
            print(f"  Right - Pred: {pred_right:.4f}, Actual: {right_mean:.4f} ± {right_std:.4f} m/s, Error: {right_error:.1f}%")
        else:
            print("  No data collected")
        
        time.sleep(0.5)

def test_straight_line(tester):
    print("\n" + "="*60)
    print("TEST 2: STRAIGHT LINE COMPENSATION")
    print("="*60)
    
    target_velocities = [0.3, 0.5, 0.7, 1.0, 1.5]
    
    for target_vel in target_velocities:
        left_pwm, right_pwm = calculate_needed_pwm(target_vel, target_vel)
        
        print(f"\n--- Target: {target_vel:.1f} m/s, PWM L={left_pwm} R={right_pwm} ---")
        data = tester.test_pwm(left_pwm, right_pwm, duration=3)
        
        if data:
            left_speeds = [abs(d['left_speed']) for d in data if abs(d['left_speed']) > 0.001]
            right_speeds = [abs(d['right_speed']) for d in data if abs(d['right_speed']) > 0.001]
            
            left_mean = np.mean(left_speeds) if left_speeds else 0
            right_mean = np.mean(right_speeds) if right_speeds else 0
            
            print(f"  Actual L: {left_mean:.3f} m/s (error: {abs(left_mean - target_vel):.3f})")
            print(f"  Actual R: {right_mean:.3f} m/s (error: {abs(right_mean - target_vel):.3f})")
        else:
            print("  No data collected")
        
        time.sleep(0.5)

def test_individual_motors(tester):
    print("\n" + "="*60)
    print("TEST 3: INDIVIDUAL MOTOR CHARACTERIZATION")
    print("="*60)
    
    print("\nLEFT motor only:")
    for pwm in [10, 20, 30, 40, 50]:
        print(f"  PWM L={pwm}, R=0...", end=" ")
        data = tester.test_pwm(pwm, 0, duration=2)
        
        if data:
            speeds = [abs(d['left_speed']) for d in data if abs(d['left_speed']) > 0.001]
            mean_speed = np.mean(speeds) if speeds else 0
            pred_speed, _ = predict_speeds(pwm, 0)
            print(f"Pred: {pred_speed:.3f}, Actual: {mean_speed:.3f} m/s")
        else:
            print("No data")
        time.sleep(0.5)
    
    print("\nRIGHT motor only:")
    for pwm in [10, 20, 30, 40, 50]:
        print(f"  PWM L=0, R={pwm}...", end=" ")
        data = tester.test_pwm(0, pwm, duration=2)
        
        if data:
            speeds = [abs(d['right_speed']) for d in data if abs(d['right_speed']) > 0.001]
            mean_speed = np.mean(speeds) if speeds else 0
            _, pred_speed = predict_speeds(0, pwm)
            print(f"Pred: {pred_speed:.3f}, Actual: {mean_speed:.3f} m/s")
        else:
            print("No data")
        time.sleep(0.5)

def test_pwm_sweep(tester):
    print("\n" + "="*60)
    print("TEST 4: PWM SWEEP")
    print("="*60)
    
    for pwm in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        print(f"  PWM={pwm}...", end=" ")
        data = tester.test_pwm(pwm, pwm, duration=2)
        
        if data:
            left_speeds = [abs(d['left_speed']) for d in data if abs(d['left_speed']) > 0.001]
            right_speeds = [abs(d['right_speed']) for d in data if abs(d['right_speed']) > 0.001]
            
            left_mean = np.mean(left_speeds) if left_speeds else 0
            right_mean = np.mean(right_speeds) if right_speeds else 0
            
            print(f"L: {left_mean:.3f}, R: {right_mean:.3f} m/s")
        else:
            print("No data")
        time.sleep(0.3)

def main():
    parser = argparse.ArgumentParser(description='Test motor PWM-velocity models')
    parser.add_argument('--port', default='COM9', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--test', choices=['all', 'accuracy', 'straight', 'individual', 'sweep'], 
                       default='all', help='Test to run')
    
    args = parser.parse_args()
    
    print("Hoverboard Motor Model Tester")
    print("==============================")
    
    tester = MotorTester(args.port, args.baud)
    
    if not tester.connect():
        print("Failed to connect!")
        sys.exit(1)
    
    try:
        if args.test in ['all', 'accuracy']:
            test_model_accuracy(tester)
        
        if args.test in ['all', 'straight']:
            test_straight_line(tester)
        
        if args.test in ['all', 'individual']:
            test_individual_motors(tester)
        
        if args.test in ['all', 'sweep']:
            test_pwm_sweep(tester)
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETE")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted")
    finally:
        tester.disconnect()
        print("Disconnected")

if __name__ == "__main__":
    main()