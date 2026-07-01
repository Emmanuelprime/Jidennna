import numpy as np
from scipy import stats, optimize, signal
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

def load_motor_data(filename):
    """Load and parse CSV data file"""
    data = defaultdict(list)
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = int(row['timestamp_ms']) / 1000.0  # Convert to seconds
            left_speed = abs(float(row['left_speed_ms']))
            right_speed = abs(float(row['right_speed_ms']))
            left_pwm = int(row['left_pwm'])
            right_pwm = int(row['right_pwm'])
            
            data['time'].append(timestamp)
            data['left_speed'].append(left_speed)
            data['right_speed'].append(right_speed)
            data['left_pwm'].append(left_pwm)
            data['right_pwm'].append(right_pwm)
    
    return data

def first_order_step_response(t, K, tau, t0=0):
    """First order step response: y(t) = K * (1 - exp(-(t-t0)/tau))"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti >= t0:
            y[i] = K * (1 - np.exp(-(ti - t0) / tau))
    return y

def fit_first_order(time, speed, pwm):
    """Fit first order model to step response data"""
    # Find when step occurs (first non-zero PWM)
    step_idx = np.where(np.array(pwm) > 0)[0]
    if len(step_idx) == 0:
        return None
    
    step_start = step_idx[0]
    
    # Use data from step start onwards
    t_data = np.array(time[step_start:]) - time[step_start]
    y_data = np.array(speed[step_start:])
    
    # Skip first few samples to avoid noise
    if len(t_data) < 10:
        return None
    
    # Initial guess
    K_guess = np.mean(y_data[-10:])  # Steady state value
    tau_guess = t_data[-1] / 3  # Time constant ~ 1/3 of total time
    
    try:
        # Fit the model
        popt, pcov = optimize.curve_fit(
            lambda t, K, tau: first_order_step_response(t, K, tau, 0),
            t_data, y_data,
            p0=[K_guess, tau_guess],
            bounds=([0, 0.01], [10, 5])
        )
        
        K, tau = popt
        
        # Calculate R²
        y_pred = first_order_step_response(t_data, K, tau, 0)
        ss_res = np.sum((y_data - y_pred)**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'K': K,
            'tau': tau,
            'r_squared': r_squared,
            't_data': t_data,
            'y_data': y_data,
            'y_pred': y_pred,
            'pwm': pwm[step_start]
        }
    except:
        return None

def analyze_all_files():
    """Analyze all CSV files and extract first-order parameters"""
    import glob
    import csv
    
    files = glob.glob("forward_*.csv")
    
    results = {
        'left': defaultdict(list),
        'right': defaultdict(list)
    }
    
    for filepath in sorted(files):
        print(f"Processing {filepath}...")
        data = load_motor_data(filepath)
        
        # Fit left motor
        left_fit = fit_first_order(data['time'], data['left_speed'], data['left_pwm'])
        if left_fit:
            pwm = left_fit['pwm']
            results['left'][pwm].append(left_fit)
            print(f"  Left:  PWM={pwm}, K={left_fit['K']:.4f}, τ={left_fit['tau']:.3f}s, R²={left_fit['r_squared']:.3f}")
        
        # Fit right motor
        right_fit = fit_first_order(data['time'], data['right_speed'], data['right_pwm'])
        if right_fit:
            pwm = right_fit['pwm']
            results['right'][pwm].append(right_fit)
            print(f"  Right: PWM={pwm}, K={right_fit['K']:.4f}, τ={right_fit['tau']:.3f}s, R²={right_fit['r_squared']:.3f}")
    
    return results

def analyze_steady_state(results):
    """Analyze steady-state gain vs PWM"""
    print("\n" + "="*60)
    print("STEADY-STATE ANALYSIS")
    print("="*60)
    
    for motor in ['left', 'right']:
        print(f"\n{motor.upper()} MOTOR:")
        print(f"{'PWM':>5} {'K (gain)':>10} {'τ (tau)':>10} {'R²':>8}")
        print("-" * 40)
        
        pwms = []
        Ks = []
        taus = []
        
        for pwm in sorted(results[motor].keys()):
            fits = results[motor][pwm]
            if fits:
                avg_K = np.mean([f['K'] for f in fits])
                avg_tau = np.mean([f['tau'] for f in fits])
                avg_r2 = np.mean([f['r_squared'] for f in fits])
                
                print(f"{pwm:5d} {avg_K:10.4f} {avg_tau:10.3f} {avg_r2:8.3f}")
                
                pwms.append(pwm)
                Ks.append(avg_K)
                taus.append(avg_tau)
        
        # Fit K vs PWM relationship
        if len(pwms) > 1:
            pwms_arr = np.array(pwms)
            Ks_arr = np.array(Ks)
            
            # Linear fit for K vs PWM
            slope, intercept, r_value, p_value, std_err = stats.linregress(pwms_arr, Ks_arr)
            print(f"\n  K(PWM) = {slope:.6f} * PWM + {intercept:.4f}")
            print(f"  R² = {r_value**2:.4f}")
            
            # Average tau
            avg_tau = np.mean(taus)
            std_tau = np.std(taus)
            print(f"  Average τ = {avg_tau:.3f} ± {std_tau:.3f} seconds")

def plot_first_order_fits(results):
    """Plot first-order fits for all PWM values"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('First-Order Motor Model Fits', fontsize=14, fontweight='bold')
    
    # Plot step responses for left motor
    ax1 = axes[0, 0]
    for pwm in sorted(results['left'].keys())[::2]:  # Plot every other PWM for clarity
        fits = results['left'][pwm]
        if fits:
            fit = fits[0]  # Use first fit
            ax1.plot(fit['t_data'], fit['y_data'], '.', alpha=0.3, markersize=2)
            ax1.plot(fit['t_data'], fit['y_pred'], '-', linewidth=2, 
                    label=f'PWM={pwm} (K={fit["K"]:.3f}, τ={fit["tau"]:.3f})')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Left Motor Step Responses')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot step responses for right motor
    ax2 = axes[0, 1]
    for pwm in sorted(results['right'].keys())[::2]:
        fits = results['right'][pwm]
        if fits:
            fit = fits[0]
            ax2.plot(fit['t_data'], fit['y_data'], '.', alpha=0.3, markersize=2)
            ax2.plot(fit['t_data'], fit['y_pred'], '-', linewidth=2,
                    label=f'PWM={pwm} (K={fit["K"]:.3f}, τ={fit["tau"]:.3f})')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Right Motor Step Responses')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot K vs PWM
    ax3 = axes[1, 0]
    for motor, color, marker in [('left', 'blue', 'o'), ('right', 'red', 's')]:
        pwms = []
        Ks = []
        for pwm in sorted(results[motor].keys()):
            fits = results[motor][pwm]
            if fits:
                pwms.append(pwm)
                Ks.append(np.mean([f['K'] for f in fits]))
        
        ax3.scatter(pwms, Ks, color=color, marker=marker, label=f'{motor.capitalize()} Motor')
        
        # Linear fit
        if len(pwms) > 1:
            slope, intercept, _, _, _ = stats.linregress(pwms, Ks)
            x_fit = np.array([min(pwms), max(pwms)])
            ax3.plot(x_fit, slope * x_fit + intercept, '--', color=color, alpha=0.7)
    
    ax3.set_xlabel('PWM Value')
    ax3.set_ylabel('Steady-State Gain K (m/s)')
    ax3.set_title('Gain vs PWM')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot tau vs PWM
    ax4 = axes[1, 1]
    for motor, color, marker in [('left', 'blue', 'o'), ('right', 'red', 's')]:
        pwms = []
        taus = []
        for pwm in sorted(results[motor].keys()):
            fits = results[motor][pwm]
            if fits:
                pwms.append(pwm)
                taus.append(np.mean([f['tau'] for f in fits]))
        
        ax4.scatter(pwms, taus, color=color, marker=marker, label=f'{motor.capitalize()} Motor')
        
        # Average tau line
        if taus:
            avg_tau = np.mean(taus)
            ax4.axhline(y=avg_tau, color=color, linestyle='--', alpha=0.7,
                       label=f'Avg τ = {avg_tau:.3f}s')
    
    ax4.set_xlabel('PWM Value')
    ax4.set_ylabel('Time Constant τ (seconds)')
    ax4.set_title('Time Constant vs PWM')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('first_order_model_fits.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_transfer_functions(results):
    """Print the transfer functions for both motors"""
    print("\n" + "="*60)
    print("TRANSFER FUNCTIONS")
    print("="*60)
    
    for motor in ['left', 'right']:
        # Calculate average K/PWM ratio and average tau
        pwms = []
        Ks = []
        taus = []
        
        for pwm in sorted(results[motor].keys()):
            fits = results[motor][pwm]
            if fits:
                pwms.append(pwm)
                Ks.append(np.mean([f['K'] for f in fits]))
                taus.append(np.mean([f['tau'] for f in fits]))
        
        if len(pwms) > 1:
            pwms_arr = np.array(pwms)
            Ks_arr = np.array(Ks)
            taus_arr = np.array(taus)
            
            # K/PWM relationship
            slope, intercept, r_value, _, _ = stats.linregress(pwms_arr, Ks_arr)
            avg_tau = np.mean(taus_arr)
            
            print(f"\n{motor.upper()} MOTOR:")
            print(f"  Continuous transfer function:")
            print(f"  G(s) = ({slope:.4f}) / ({avg_tau:.3f}s + 1)")
            print(f"  G(s) = {slope/avg_tau:.4f} / (s + {1/avg_tau:.4f})")
            print(f"  Where K = {slope:.6f} * PWM + {intercept:.4f}")
            print(f"  Average time constant τ = {avg_tau:.3f} seconds")
            
            # Discretize (assuming 100ms sample time typical for control)
            Ts = 0.1
            print(f"\n  Discrete transfer function (Ts={Ts}s, Zero-Order Hold):")
            
            # G(z) = K * (1 - exp(-Ts/tau)) / (z - exp(-Ts/tau))
            a = np.exp(-Ts / avg_tau)
            b = slope * (1 - a)
            print(f"  G(z) = {b:.4f} / (z - {a:.4f})")
            print(f"  Difference equation: y[k] = {a:.4f}*y[k-1] + {b:.4f}*u[k-1]")

def main():
    import csv
    
    print("First-Order Motor Model Identification")
    print("======================================")
    
    results = analyze_all_files()
    
    if not results['left'] and not results['right']:
        print("No data found!")
        return
    
    analyze_steady_state(results)
    print_transfer_functions(results)
    plot_first_order_fits(results)

if __name__ == "__main__":
    main()