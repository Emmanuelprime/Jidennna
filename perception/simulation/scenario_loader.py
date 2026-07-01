"""
Load and manage simulation scenarios from YAML files.
"""

import yaml
import os
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ScenarioLoader:
    """Loads simulation scenarios from configuration files"""
    
    def __init__(self, scenario_dir: str = None):
        self.scenario_dir = scenario_dir or os.path.join(
            os.path.dirname(__file__), '..', 'config', 'simulation_scenarios'
        )
        self.current_scenario = None
    
    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load a scenario by name
        
        Args:
            scenario_name: Name of the scenario file (without .yaml)
            
        Returns:
            Scenario configuration dictionary
        """
        scenario_path = os.path.join(self.scenario_dir, f"{scenario_name}.yaml")
        
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario not found: {scenario_path}")
        
        with open(scenario_path, 'r') as f:
            scenario = yaml.safe_load(f)
        
        self.current_scenario = scenario
        logger.info(f"Loaded scenario: {scenario_name}")
        return scenario
    
    def get_pedestrians(self) -> List[Dict]:
        """Get pedestrian configurations from current scenario"""
        if not self.current_scenario:
            return []
        return self.current_scenario.get('pedestrians', [])
    
    def get_vehicles(self) -> List[Dict]:
        """Get vehicle configurations from current scenario"""
        if not self.current_scenario:
            return []
        return self.current_scenario.get('vehicles', [])
    
    def get_static_obstacles(self) -> List[Dict]:
        """Get static obstacle configurations"""
        if not self.current_scenario:
            return []
        return self.current_scenario.get('static_obstacles', [])
    
    def get_robot_start(self) -> Dict:
        """Get robot starting configuration"""
        if not self.current_scenario:
            return {'position': [0, 0, 0]}
        return self.current_scenario.get('robot_start', {'position': [0, 0, 0]})
    
    def get_environment(self) -> Dict:
        """Get environment configuration"""
        if not self.current_scenario:
            return {}
        return self.current_scenario.get('environment', {})
    
    def list_scenarios(self) -> List[str]:
        """List all available scenarios"""
        scenarios = []
        if os.path.exists(self.scenario_dir):
            for file in os.listdir(self.scenario_dir):
                if file.endswith('.yaml'):
                    scenarios.append(file[:-5])
        return scenarios
    
    def create_default_scenario(self, name: str) -> Dict:
        """Create a default scenario configuration"""
        scenario = {
            'name': name,
            'description': 'Default simulation scenario',
            'robot_start': {
                'position': [0.0, 0.0, 0.0]
            },
            'environment': {
                'size': [50.0, 50.0],
                'ground_type': 'flat'
            },
            'pedestrians': [
                {
                    'start_position': [5.0, 2.0],
                    'goal_position': [-5.0, -2.0],
                    'speed': 1.4
                },
                {
                    'start_position': [-3.0, 4.0],
                    'goal_position': [3.0, -4.0],
                    'speed': 1.2
                }
            ],
            'vehicles': [
                {
                    'position': [10.0, 0.0],
                    'velocity': [-2.0, 0.0],
                    'path': [[10.0, 0.0], [-10.0, 0.0]]
                }
            ],
            'static_obstacles': [
                {
                    'position': [2.0, 3.0],
                    'radius': 0.5
                },
                {
                    'position': [-2.0, -3.0],
                    'radius': 0.3
                }
            ]
        }
        
        # Save scenario
        scenario_path = os.path.join(self.scenario_dir, f"{name}.yaml")
        os.makedirs(self.scenario_dir, exist_ok=True)
        
        with open(scenario_path, 'w') as f:
            yaml.dump(scenario, f, default_flow_style=False)
        
        logger.info(f"Created default scenario: {scenario_path}")
        return scenario