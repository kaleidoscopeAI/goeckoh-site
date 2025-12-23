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

