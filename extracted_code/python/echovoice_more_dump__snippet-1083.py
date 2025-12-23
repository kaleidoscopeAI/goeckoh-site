def __init__(self):
    self.map = {
        "conserve_energy": [{"op":"no_op", "delay": 0.0}],
        "ingest_new_data": [{"op":"open_url", "url":"https://example.com", "delay":0.1}],
        "defensive_stance_disconnect": [{"op":"network_disable", "delay":0.01}],
        "type_text": [{"op":"key_press","key":"t","delay":0.05}, {"op":"key_press","key":"e","delay":0.05}, {"op":"key_press","key":"s","delay":0.05}, {"op":"key_press","key":"t","delay":0.05}],
    }

def get_sequence(self, intent_name: str, params: Dict=None) -> (List[Dict], float):
    params = params or {}
    seq = self.map.get(intent_name, [{"op":"no_op"}])
    cost = params.get("cost", len(seq) * 0.1)  # Real cost based on seq length
    return seq, cost

