import serial
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import threading

class MotorCharacterizer:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.data = {
            'timestamp': [],
            'left_pulses': [],
            'left_speed': [],
            'right_pulses': [],
            'right_speed': [],
            'left_pwm': [],
            'right_pwm': [],
            'command': []
        }
        self.running = False
        self.collecting = False
        self.current_left_pwm = 0
        self.current_right_pwm = 0
        self.current_cmd = 'idle'
        
    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(3)
            self.ser.reset_input_buffer()
            print(f"Connected to {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def send_command(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write((cmd + '\n').encode())
            time.sleep(0.05)
            return True
        return False
    
    def start_data_collection(self):
        self.collecting = True
        self.data = {key: [] for key in self.data}
        self.data_thread = threading.Thread(target=self._collect_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        print("Data collection started")
    
    def stop_data_collection(self):
        self.collecting = False
        if hasattr(self, 'data_thread'):
            self.data_thread.join(timeout=2)
        print("Data collection stopped")
    
    def _collect_data(self):
        while self.collecting:
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith('CNT,'):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            timestamp = int(parts[1])
                            left_pulses = int(parts[2])
                            left_speed = float(parts[3])
                            right_pulses = int(parts[4])
                            right_speed = float(parts[5])
                            
                            self.data['timestamp'].append(timestamp)
                            self.data['left_pulses'].append(left_pulses)
                            self.data['left_speed'].append(left_speed)
                            self.data['right_pulses'].append(right_pulses)
                            self.data['right_speed'].append(right_speed)
                            self.data['left_pwm'].append(self.current_left_pwm)
                            self.data['right_pwm'].append(self.current_right_pwm)
                            self.data['command'].append(self.current_cmd)
                            
                        except (ValueError, IndexError) as e:
                            pass
                elif line.startswith('#') or line.startswith('READY'):
                    print(f"Arduino: {line}")
            else:
                time.sleep(0.001)
    
    def pwm_step_test(self, pwm_values=None, hold_time=5, settle_time=1):
        """Run a stepped PWM test with settle time before collecting data"""
        if pwm_values is None:
            pwm_values = list(range(5, 61, 5))  # More granular: 5,10,15,...,60
        
        print(f"Running PWM step test: {pwm_values}")
        
        self.send_command('s')
        time.sleep(2)
        
        self.start_data_collection()
        time.sleep(1)
        
        for pwm in pwm_values:
            print(f"\nSetting PWM to {pwm}")
            self.current_left_pwm = pwm
            self.current_right_pwm = pwm
            self.current_cmd = f'f{pwm}'
            
            self.send_command(f'f{pwm}')
            
            # Wait for motor to reach steady state before collecting data
            print(f"Waiting {settle_time}s to settle, then {hold_time}s to collect data...")
            time.sleep(settle_time)
            # The data collection continues during the hold time
            time.sleep(hold_time)
        
        self.send_command('s')
        self.current_cmd = 'stop'
        time.sleep(1)
        self.stop_data_collection()
        
        print(f"\nTest complete. Collected {len(self.data['left_speed'])} data points")
    
    def analyze_data(self):
        """Enhanced analysis with motor mismatch and nonlinearity detection"""
        if len(self.data['left_speed']) == 0:
            print("No data collected")
            return None, None
        
        df = pd.DataFrame(self.data)
        
        # Filter out idle periods and unstable data
        df_filtered = df[(df['left_speed'] > 0.02) & (df['right_speed'] > 0.02)]
        
        if len(df_filtered) == 0:
            print("No significant movement detected")
            return None, None
        
        # Calculate statistics for each PWM value
        left_avg = df_filtered.groupby('left_pwm')['left_speed'].agg(['mean', 'std', 'count']).reset_index()
        right_avg = df_filtered.groupby('right_pwm')['right_speed'].agg(['mean', 'std', 'count']).reset_index()
        
        print("\n" + "="*60)
        print("MOTOR CHARACTERIZATION RESULTS")
        print("="*60)
        
        # Left motor analysis
        if len(left_avg) > 2:
            left_slope, left_intercept, left_r, left_p, left_std_err = stats.linregress(
                left_avg['left_pwm'], left_avg['mean']
            )
            print(f"\nLEFT MOTOR:")
            print(f"  Linear fit: v = {left_slope:.4f} * PWM + {left_intercept:.4f}")
            print(f"  R² = {left_r**2:.4f}")
            print(f"  Max speed: {left_avg['mean'].max():.3f} m/s at PWM {left_avg.loc[left_avg['mean'].idxmax(), 'left_pwm']}")
            
            # Detect saturation (if max speed is at highest PWM, motor may not be saturated yet)
            if left_avg.loc[left_avg['mean'].idxmax(), 'left_pwm'] == left_avg['left_pwm'].max():
                print("  ⚠️  Motor may not be saturated - max speed at highest PWM")
        
        # Right motor analysis
        if len(right_avg) > 2:
            right_slope, right_intercept, right_r, right_p, right_std_err = stats.linregress(
                right_avg['right_pwm'], right_avg['mean']
            )
            print(f"\nRIGHT MOTOR:")
            print(f"  Linear fit: v = {right_slope:.4f} * PWM + {right_intercept:.4f}")
            print(f"  R² = {right_r**2:.4f}")
            print(f"  Max speed: {right_avg['mean'].max():.3f} m/s at PWM {right_avg.loc[right_avg['mean'].idxmax(), 'right_pwm']}")
        
        # Motor comparison
        if len(left_avg) > 2 and len(right_avg) > 2:
            print(f"\nMOTOR MISMATCH:")
            # Calculate mismatch at each PWM value
            left_avg_sorted = left_avg.sort_values('left_pwm')
            right_avg_sorted = right_avg.sort_values('right_pwm')
            
            # Interpolate to compare at same PWM values
            common_pwms = np.intersect1d(left_avg_sorted['left_pwm'], right_avg_sorted['right_pwm'])
            if len(common_pwms) > 0:
                left_speeds = left_avg_sorted[left_avg_sorted['left_pwm'].isin(common_pwms)]['mean'].values
                right_speeds = right_avg_sorted[right_avg_sorted['right_pwm'].isin(common_pwms)]['mean'].values
                
                speed_ratios = left_speeds / right_speeds
                avg_ratio = np.mean(speed_ratios)
                print(f"  Left/Right speed ratio: {avg_ratio:.2f} (Left is {avg_ratio*100:.0f}% of Right)")
                print(f"  ⚠️  Significant motor mismatch detected - need to compensate in software")
            
            # Suggest compensation
            print(f"\nRECOMMENDED COMPENSATION:")
            print(f"  Scale factor for Right motor: {left_slope/right_slope:.3f}")
            print(f"  To match speeds: multiply Right PWM by {left_slope/right_slope:.3f}")
        
        print("\n" + "="*60)
        
        return left_avg, right_avg
    
    def plot_results(self):
        """Enhanced plotting with more insights"""
        if len(self.data['left_speed']) == 0:
            print("No data to plot")
            return
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        left_df = pd.DataFrame(self.data)
        left_filtered = left_df[left_df['left_speed'] > 0.02]
        right_filtered = left_df[left_df['right_speed'] > 0.02]
        
        # 1. Left motor speed vs PWM
        ax1 = fig.add_subplot(gs[0, 0])
        left_avg = left_filtered.groupby('left_pwm')['left_speed'].mean().reset_index()
        left_std = left_filtered.groupby('left_pwm')['left_speed'].std().reset_index()
        
        ax1.scatter(self.data['left_pwm'], self.data['left_speed'], alpha=0.2, label='Raw data', s=2)
        if len(left_avg) > 2:
            ax1.plot(left_avg['left_pwm'], left_avg['left_speed'], 'r-', linewidth=2, label='Mean')
            ax1.fill_between(left_avg['left_pwm'], 
                           left_avg['left_speed'] - left_std['left_speed'],
                           left_avg['left_speed'] + left_std['left_speed'],
                           alpha=0.2, color='r', label='±1 std')
            z = np.polyfit(left_avg['left_pwm'], left_avg['left_speed'], 1)
            p = np.poly1d(z)
            ax1.plot(left_avg['left_pwm'], p(left_avg['left_pwm']), 'g--', label=f'Linear: v={z[0]:.3f}PWM+{z[1]:.3f}')
        ax1.set_xlabel('PWM')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Left Motor')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Right motor speed vs PWM
        ax2 = fig.add_subplot(gs[0, 1])
        right_avg = right_filtered.groupby('right_pwm')['right_speed'].mean().reset_index()
        right_std = right_filtered.groupby('right_pwm')['right_speed'].std().reset_index()
        
        ax2.scatter(self.data['right_pwm'], self.data['right_speed'], alpha=0.2, label='Raw data', s=2)
        if len(right_avg) > 2:
            ax2.plot(right_avg['right_pwm'], right_avg['right_speed'], 'r-', linewidth=2, label='Mean')
            ax2.fill_between(right_avg['right_pwm'],
                           right_avg['right_speed'] - right_std['right_speed'],
                           right_avg['right_speed'] + right_std['right_speed'],
                           alpha=0.2, color='r', label='±1 std')
            z = np.polyfit(right_avg['right_pwm'], right_avg['right_speed'], 1)
            p = np.poly1d(z)
            ax2.plot(right_avg['right_pwm'], p(right_avg['right_pwm']), 'g--', label=f'Linear: v={z[0]:.3f}PWM+{z[1]:.3f}')
        ax2.set_xlabel('PWM')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Right Motor')
        ax2.grid(True)
        ax2.legend()
        
        # 3. Motor comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if len(left_avg) > 2 and len(right_avg) > 2:
            ax3.plot(left_avg['left_pwm'], left_avg['left_speed'], 'b-o', label='Left', linewidth=2, markersize=6)
            ax3.plot(right_avg['right_pwm'], right_avg['right_speed'], 'r-o', label='Right', linewidth=2, markersize=6)
            
            # Add speed ratio
            common_pwms = np.intersect1d(left_avg['left_pwm'], right_avg['right_pwm'])
            if len(common_pwms) > 0:
                left_speeds = left_avg[left_avg['left_pwm'].isin(common_pwms)]['left_speed'].values
                right_speeds = right_avg[right_avg['right_pwm'].isin(common_pwms)]['right_speed'].values
                ratio = left_speeds / right_speeds
                for i, pwm in enumerate(common_pwms):
                    ax3.annotate(f'{ratio[i]:.2f}x', (pwm, max(left_speeds[i], right_speeds[i])),
                               textcoords="offset points", xytext=(0,10), ha='center')
            
        ax3.set_xlabel('PWM')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Motor Comparison (values show Left/Right ratio)')
        ax3.grid(True)
        ax3.legend()
        
        # 4. Speed over time (showing test steps)
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax4.plot(self.data['timestamp'], self.data['left_speed'], label='Left', alpha=0.7)
        ax4.plot(self.data['timestamp'], self.data['right_speed'], label='Right', alpha=0.7)
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Speed vs Time (shows PWM steps)')
        ax4.grid(True)
        ax4.legend()
        
        # 5. Scatter plot of speed difference
        ax5 = fig.add_subplot(gs[1, 2])
        speed_diff = np.array(self.data['left_speed']) - np.array(self.data['right_speed'])
        ax5.scatter(self.data['left_pwm'], speed_diff, alpha=0.3, s=2)
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('PWM')
        ax5.set_ylabel('Speed Difference (Left - Right) [m/s]')
        ax5.set_title('Motor Mismatch vs PWM')
        ax5.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_data(self, filename='motor_data.csv'):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    def debug_serial(self, duration=5):
        print(f"Debugging serial for {duration} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"RAW: {line}")
            time.sleep(0.01)
    
    def stop(self):
        self.collecting = False
        self.running = False
        if self.ser and self.ser.is_open:
            self.send_command('s')
            time.sleep(0.5)
            self.ser.close()
            print("Connection closed")

def main():
    PORT = 'COM9'
    CHARACTERIZER = MotorCharacterizer(PORT)
    
    try:
        if not CHARACTERIZER.connect():
            return
        
        print("\n=== Starting Step Test ===")
        CHARACTERIZER.pwm_step_test(pwm_values=range(10, 61, 5), hold_time=2, settle_time=1)
        
        CHARACTERIZER.analyze_data()
        CHARACTERIZER.plot_results()
        CHARACTERIZER.save_data('motor_characterization_detailed.csv')
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        CHARACTERIZER.stop()

if __name__ == "__main__":
    main()