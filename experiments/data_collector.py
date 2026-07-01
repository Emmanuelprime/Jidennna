import serial
import time
import csv
import argparse
import sys
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Collect hoverboard motor data')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--f', type=int, help='Forward PWM value (0-60)')
    parser.add_argument('--r', type=int, help='Reverse PWM value (0-60)')
    parser.add_argument('--duration', type=int, default=10, help='Collection duration in seconds (default: 10)')
    parser.add_argument('--output', help='Output CSV file (default: auto-generated)')
    
    args = parser.parse_args()
    
    if args.f is None and args.r is None:
        print("Error: Must specify either --f or --r")
        sys.exit(1)
    
    if args.f is not None and args.r is not None:
        print("Error: Cannot specify both --f and --r")
        sys.exit(1)
    
    pwm_value = args.f if args.f is not None else args.r
    direction = "forward" if args.f is not None else "reverse"
    
    if args.output:
        filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{direction}_{pwm_value}_{timestamp}.csv"
    
    print(f"Connecting to {args.port} at {args.baud} baud...")
    
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)
    
    ser.reset_input_buffer()
    
    print(f"Starting data collection: {direction} at PWM {pwm_value}")
    print(f"Duration: {args.duration} seconds")
    print(f"Output file: {filename}")
    
    command = f"STREAM:{pwm_value},{pwm_value}" if direction == "forward" else f"STREAM:{-pwm_value},{-pwm_value}"
    ser.write(f"{command}\n".encode())
    
    time.sleep(0.5)
    
    if ser.in_waiting:
        response = ser.readline().decode().strip()
        if "STREAMING" not in response:
            print(f"Error: Unexpected response: {response}")
            ser.close()
            sys.exit(1)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp_ms', 'left_count', 'right_count', 
                         'left_speed_ms', 'right_speed_ms', 
                         'left_pwm', 'right_pwm', 
                         'left_direction', 'right_direction'])
        
        start_time = time.time()
        line_count = 0
        
        try:
            while time.time() - start_time < args.duration:
                if ser.in_waiting:
                    line = ser.readline().decode().strip()
                    
                    if line.startswith("DATA,"):
                        parts = line.split(',')
                        if len(parts) == 10:
                            writer.writerow(parts[1:])
                            line_count += 1
                            
                            if line_count % 10 == 0:
                                elapsed = time.time() - start_time
                                print(f"Collected {line_count} samples ({elapsed:.1f}s elapsed)")
                
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
    
    ser.write(b"STOP_STREAM\n")
    time.sleep(0.1)
    ser.write(b"s\n")
    ser.close()
    
    print(f"\nCollection complete!")
    print(f"Total samples: {line_count}")
    print(f"Data saved to: {filename}")

if __name__ == "__main__":
    main()