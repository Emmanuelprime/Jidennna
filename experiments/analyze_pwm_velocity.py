import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

def load_and_process_files(directory="."):
    forward_files = glob.glob(os.path.join(directory, "forward_*.csv"))
    
    if not forward_files:
        print("No forward_*.csv files found!")
        return None, None, None
    
    print(f"Found {len(forward_files)} forward files:")
    for f in forward_files:
        print(f"  - {f}")
    
    left_data = defaultdict(list)
    right_data = defaultdict(list)
    
    for filepath in forward_files:
        filename = os.path.basename(filepath)
        pwm_value = int(filename.split('_')[1])
        
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            left_speeds = []
            right_speeds = []
            
            for row in reader:
                left_speed = float(row['left_speed_ms'])
                right_speed = float(row['right_speed_ms'])
                
                if abs(left_speed) > 0.001:
                    left_speeds.append(abs(left_speed))
                if abs(right_speed) > 0.001:
                    right_speeds.append(abs(right_speed))
            
            if left_speeds:
                mean_left_speed = np.mean(left_speeds)
                std_left_speed = np.std(left_speeds)
                left_data[pwm_value].append({
                    'mean': mean_left_speed,
                    'std': std_left_speed,
                    'count': len(left_speeds)
                })
            
            if right_speeds:
                mean_right_speed = np.mean(right_speeds)
                std_right_speed = np.std(right_speeds)
                right_data[pwm_value].append({
                    'mean': mean_right_speed,
                    'std': std_right_speed,
                    'count': len(right_speeds)
                })
    
    return left_data, right_data, forward_files

def calculate_statistics(data_dict):
    pwms = []
    means = []
    stds = []
    
    for pwm in sorted(data_dict.keys()):
        values = data_dict[pwm]
        if values:
            pwms.append(pwm)
            combined_mean = np.mean([v['mean'] for v in values])
            combined_std = np.mean([v['std'] for v in values])
            means.append(combined_mean)
            stds.append(combined_std)
    
    return np.array(pwms), np.array(means), np.array(stds)

def fit_relationships(pwms, means):
    results = {}
    
    if len(pwms) < 2:
        return results
    
    # Linear fit (y = mx + b)
    slope, intercept, r_value, p_value, std_err = stats.linregress(pwms, means)
    results['linear'] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'equation': f'v = {slope:.4f} * PWM + {intercept:.4f}'
    }
    
    # Quadratic fit (y = ax² + bx + c)
    try:
        coeffs = np.polyfit(pwms, means, 2)
        poly = np.poly1d(coeffs)
        predicted = poly(pwms)
        ss_res = np.sum((means - predicted)**2)
        ss_tot = np.sum((means - np.mean(means))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results['quadratic'] = {
            'coefficients': coeffs,
            'r_squared': r_squared,
            'equation': f'v = {coeffs[0]:.6f} * PWM² + {coeffs[1]:.4f} * PWM + {coeffs[2]:.4f}'
        }
    except:
        pass
    
    # Power law fit (y = ax^b)
    try:
        log_pwms = np.log(pwms[pwms > 0])
        log_means = np.log(means[pwms > 0])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_pwms, log_means)
        a = np.exp(intercept)
        b = slope
        
        predicted = a * pwms**b
        ss_res = np.sum((means - predicted)**2)
        ss_tot = np.sum((means - np.mean(means))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results['power'] = {
            'a': a,
            'b': b,
            'r_squared': r_squared,
            'equation': f'v = {a:.4f} * PWM^{b:.4f}'
        }
    except:
        pass
    
    return results

def plot_results(left_pwms, left_means, left_stds, right_pwms, right_means, right_stds, 
                 left_fits, right_fits, forward_files):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PWM-Velocity Relationship Analysis', fontsize=14, fontweight='bold')
    
    # Left motor plot
    ax1 = axes[0, 0]
    ax1.errorbar(left_pwms, left_means, yerr=left_stds, fmt='bo-', capsize=5, label='Left Motor')
    
    if left_fits and 'linear' in left_fits:
        fit = left_fits['linear']
        x_fit = np.linspace(min(left_pwms), max(left_pwms), 100)
        y_fit = fit['slope'] * x_fit + fit['intercept']
        ax1.plot(x_fit, y_fit, 'r--', alpha=0.7, 
                label=f'Linear fit (R²={fit["r_squared"]:.3f})')
    
    if left_fits and 'quadratic' in left_fits:
        fit = left_fits['quadratic']
        coeffs = fit['coefficients']
        x_fit = np.linspace(min(left_pwms), max(left_pwms), 100)
        y_fit = coeffs[0]*x_fit**2 + coeffs[1]*x_fit + coeffs[2]
        ax1.plot(x_fit, y_fit, 'g--', alpha=0.7, 
                label=f'Quadratic fit (R²={fit["r_squared"]:.3f})')
    
    ax1.set_xlabel('PWM Value')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Left Motor PWM-Velocity Relationship')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right motor plot
    ax2 = axes[0, 1]
    ax2.errorbar(right_pwms, right_means, yerr=right_stds, fmt='ro-', capsize=5, label='Right Motor')
    
    if right_fits and 'linear' in right_fits:
        fit = right_fits['linear']
        x_fit = np.linspace(min(right_pwms), max(right_pwms), 100)
        y_fit = fit['slope'] * x_fit + fit['intercept']
        ax2.plot(x_fit, y_fit, 'b--', alpha=0.7, 
                label=f'Linear fit (R²={fit["r_squared"]:.3f})')
    
    if right_fits and 'quadratic' in right_fits:
        fit = right_fits['quadratic']
        coeffs = fit['coefficients']
        x_fit = np.linspace(min(right_pwms), max(right_pwms), 100)
        y_fit = coeffs[0]*x_fit**2 + coeffs[1]*x_fit + coeffs[2]
        ax2.plot(x_fit, y_fit, 'm--', alpha=0.7, 
                label=f'Quadratic fit (R²={fit["r_squared"]:.3f})')
    
    ax2.set_xlabel('PWM Value')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Right Motor PWM-Velocity Relationship')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Comparison plot
    ax3 = axes[1, 0]
    ax3.errorbar(left_pwms, left_means, yerr=left_stds, fmt='bo-', capsize=5, 
                label='Left Motor', alpha=0.7)
    ax3.errorbar(right_pwms, right_means, yerr=right_stds, fmt='ro-', capsize=5, 
                label='Right Motor', alpha=0.7)
    
    ax3.set_xlabel('PWM Value')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Left vs Right Motor Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    text_lines = ["RELATIONSHIP ANALYSIS RESULTS\n"]
    
    if left_fits and right_fits:
        text_lines.append("LEFT MOTOR:")
        for fit_type, fit_data in left_fits.items():
            text_lines.append(f"  {fit_type.capitalize()}: {fit_data['equation']}")
            text_lines.append(f"    R² = {fit_data['r_squared']:.4f}\n")
        
        text_lines.append("RIGHT MOTOR:")
        for fit_type, fit_data in right_fits.items():
            text_lines.append(f"  {fit_type.capitalize()}: {fit_data['equation']}")
            text_lines.append(f"    R² = {fit_data['r_squared']:.4f}\n")
        
        if 'linear' in left_fits and 'linear' in right_fits:
            l_r2 = left_fits['linear']['r_squared']
            r_r2 = right_fits['linear']['r_squared']
            
            if l_r2 > 0.95 and r_r2 > 0.95:
                text_lines.append("CONCLUSION: Both motors show highly linear")
                text_lines.append("PWM-velocity relationships.")
            elif l_r2 > 0.9 and r_r2 > 0.9:
                text_lines.append("CONCLUSION: Both motors show good linear")
                text_lines.append("relationships with some deviation.")
            else:
                best_left = max(left_fits.items(), key=lambda x: x[1]['r_squared'])
                best_right = max(right_fits.items(), key=lambda x: x[1]['r_squared'])
                text_lines.append(f"CONCLUSION: Non-linear relationships detected.")
                text_lines.append(f"Best left fit: {best_left[0]} (R²={best_left[1]['r_squared']:.3f})")
                text_lines.append(f"Best right fit: {best_right[0]} (R²={best_right[1]['r_squared']:.3f})")
    
    ax4.text(0.1, 0.9, '\n'.join(text_lines), transform=ax4.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pwm_velocity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    left_data, right_data, files = load_and_process_files()
    
    if left_data is None:
        return
    
    left_pwms, left_means, left_stds = calculate_statistics(left_data)
    right_pwms, right_means, right_stds = calculate_statistics(right_data)
    
    print(f"\nLeft Motor Data Points: {len(left_pwms)}")
    for pwm, mean, std in zip(left_pwms, left_means, left_stds):
        print(f"  PWM {pwm:2d}: {mean:.4f} ± {std:.4f} m/s")
    
    print(f"\nRight Motor Data Points: {len(right_pwms)}")
    for pwm, mean, std in zip(right_pwms, right_means, right_stds):
        print(f"  PWM {pwm:2d}: {mean:.4f} ± {std:.4f} m/s")
    
    print("\nFitting relationships...")
    left_fits = fit_relationships(left_pwms, left_means)
    right_fits = fit_relationships(right_pwms, right_means)
    
    print("\nLeft Motor Relationships:")
    for fit_type, fit_data in left_fits.items():
        print(f"  {fit_type.capitalize()}: {fit_data['equation']} (R²={fit_data['r_squared']:.4f})")
    
    print("\nRight Motor Relationships:")
    for fit_type, fit_data in right_fits.items():
        print(f"  {fit_type.capitalize()}: {fit_data['equation']} (R²={fit_data['r_squared']:.4f})")
    
    plot_results(left_pwms, left_means, left_stds, right_pwms, right_means, right_stds,
                left_fits, right_fits, files)

if __name__ == "__main__":
    main()