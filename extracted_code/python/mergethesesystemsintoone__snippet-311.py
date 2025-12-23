import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import time
import random

class KaleidoscopeEngine:
    """
    Processes data through a series of transformations, simulating the
    kaleidoscope's intricate refractions and reflections, to generate
    refined insights.
    """

    def __init__(self, num_gears: int = 5):
        self.num_gears = num_gears
        self.gears = [Gear() for _ in range(num_gears)]
        self.gear_connections = self._initialize_gear_connections()
        self.insight_history = []

    def _initialize_gear_connections(self) -> Dict[int, List[int]]:
        """
        Establishes connections between gears.

        Returns:
            dict: A dictionary representing connections between gears.
        """
        connections = defaultdict(list)
        for i in range(self.num_gears):
            num_connections = random.randint(1, 3)  # Each gear connects to 1-3 others
            connected_gears = random.sample(
                [g for g in range(self.num_gears) if g != i],
                num_connections
            )
            connections[i].extend(connected_gears)
        return connections

    def process_data(self, data_chunk: Any) -> Dict[str, Any]:
        """
        Processes a data chunk through the series of interconnected gears.

        Args:
            data_chunk: The data chunk to be processed.

        Returns:
            dict: The processed data with added insights.
        """
        current_gear_index = 0  # Start from the first gear
        processed_data = data_chunk
        history = []

        for _ in range(self.num_gears):
            gear = self.gears[current_gear_index]
            processed_data = gear.process(processed_data)
            history.append({
                'gear_index': current_gear_index,
                'data': processed_data
            })

            # Move to the next connected gear
            connected_gears = self.gear_connections.get(current_gear_index, [])
            if connected_gears:
                current_gear_index = random.choice(connected_gears)
            else:
                break  # No further connections

        insights = self._generate_insights(processed_data)
        self.insight_history.append(insights)

        return {
            "processed_data": processed_data,
            "insights": insights,
            "processing_history": history
        }

    def _generate_insights(self, data: Any) -> Dict[str, Any]:
        """
        Generates insights based on the data processed by the gears.

        Args:
            data: The processed data.

        Returns:
            dict: Generated insights.
        """
        # Basic insight generation based on the length of the data
        insight = {
            'timestamp': time.time(),
            'data_length': len(data) if isinstance(data, (list, str)) else 0,
            'data_type': str(type(data)),
            'pattern_detected': 'complex' if len(data) > 10 else 'simple'
        }
        return insight

    def get_gear_states(self) -> List[Dict[str, Any]]:
        """
        Returns the current state of all gears in the engine.

        Returns:
            list: A list of dictionaries, each representing a gear's state.
        """
        return [gear.get_state() for gear in self.gears]

class Gear:
    """
    Represents a single gear in the Kaleidoscope Engine, capable of
    transforming data in unique ways.
    """
    def __init__(self):
        self.rotation = 0
        self.transformation_matrix = self._initialize_transformation_matrix()

    def _initialize_transformation_matrix(self) -> np.ndarray:
        """
        Initializes a transformation matrix with random values.

        Returns:
            np.ndarray: A 2x2 transformation matrix.
        """
        return np.random.rand(2, 2)

    def process(self, data: Any) -> Any:
        """
        Transforms the input data based on the gear's current state.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        self.rotate()

        if isinstance(data, list):
            transformed_data = [self._transform_value(item) for item in data]
        elif isinstance(data, dict):
            transformed_data = {k: self._transform_value(v) for k, v in data.items()}
        else:
            transformed_data = self._transform_value(data)

        return transformed_data

    def _transform_value(self, value: Any) -> Any:
        """
        Applies a transformation to a single data value.

        Args:
            value: The value to be transformed.

        Returns:
            The transformed value.
        """
        if isinstance(value, (int, float)):
            # Apply a simple transformation for numerical values
            return value * np.random.uniform(0.8, 1.2)
        elif isinstance(value, str):
            # Reverse the string as a basic transformation for text
            return value[::-1]
        else:
            return value  # Return unchanged if not a supported type

    def rotate(self):
        """
        Rotates the gear, changing its transformation behavior.
        """
        self.rotation = (self.rotation + np.random.randint(1, 46)) % 360

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the gear.

        Returns:
            dict: A dictionary containing the gear's rotation and transformation matrix.
        """
        return {
            'rotation': self.rotation,
            'transformation_matrix': self.transformation_matrix.tolist()
        }

