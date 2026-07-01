"""
Layer 4: Control Interface - Convert MPC outputs to wheel commands
"""
import numpy as np
from config.robot_params import RobotParams
from layers.layer1_plant_model import DifferentialDriveModel

class ControlInterface:
    """
    Interface between MPC controller and motor commands
    """
    
    def __init__(self):
        self.model = DifferentialDriveModel()
        self.max_rpm = 200  # Maximum wheel RPM
        
    def convert_to_wheel_commands(self, v, omega):
        """
        Convert robot velocities to wheel RPM commands
        
        Args:
            v: linear velocity (m/s)
            omega: angular velocity (rad/s)
            
        Returns:
            rpm_left, rpm_right: wheel RPM commands
        """
        # Convert to wheel velocities
        v_left, v_right = self.model.inverse_kinematics(v, omega)
        
        # Convert to RPM
        rpm_left, rpm_right = self.model.wheel_velocities_to_rpm(v_left, v_right)
        
        # Saturate RPM commands
        rpm_left = np.clip(rpm_left, -self.max_rpm, self.max_rpm)
        rpm_right = np.clip(rpm_right, -self.max_rpm, self.max_rpm)
        
        return rpm_left, rpm_right
    
    def convert_from_wheel_feedback(self, rpm_left, rpm_right):
        """
        Convert wheel RPM feedback to robot velocities
        
        Args:
            rpm_left, rpm_right: measured wheel RPM
            
        Returns:
            v, omega: estimated robot velocities
        """
        return self.model.forward_kinematics_from_wheels(rpm_left, rpm_right)
    
    def generate_control_packet(self, v, omega, sequence_number):
        """
        Generate control packet for ESP32
        
        Args:
            v: linear velocity (m/s)
            omega: angular velocity (rad/s)
            sequence_number: packet sequence number
            
        Returns:
            packet: bytes to send
        """
        rpm_left, rpm_right = self.convert_to_wheel_commands(v, omega)
        
        # Packet format: [start_byte, seq_num, rpm_left(2), rpm_right(2), checksum]
        packet = bytearray()
        packet.append(0xAA)  # Start byte
        packet.append(sequence_number & 0xFF)
        
        # RPM values as 16-bit integers (scaled by 100 for precision)
        rpm_left_int = int(rpm_left * 100)
        rpm_right_int = int(rpm_right * 100)
        
        packet.extend(rpm_left_int.to_bytes(2, byteorder='little', signed=True))
        packet.extend(rpm_right_int.to_bytes(2, byteorder='little', signed=True))
        
        # Simple checksum
        checksum = sum(packet[1:]) & 0xFF
        packet.append(checksum)
        
        return bytes(packet)