class KaleidoscopeEngine:
    logic_weights = {"gear_1": 1.0, "gear_2": 0.5, "gear_3": 0.25}

    @staticmethod
    def process_data_dumps(data_dumps: List[Dict]) -> List[Dict]:
        """Process data dumps and adjust logic weights."""
        insights = []
        total_weight = sum(KaleidoscopeEngine.logic_weights.values())

        for dump in data_dumps:
            KaleidoscopeEngine.logic_weights["gear_1"] += 0.1 * dump["data"]
            KaleidoscopeEngine.logic_weights["gear_2"] *= np.exp(-0.05 * dump["data"])
            KaleidoscopeEngine.logic_weights["gear_3"] += 0.02 * dump["data"]

            # Normalize weights to keep them bounded
            total_weight = sum(KaleidoscopeEngine.logic_weights.values())
            if total_weight > 0:
                KaleidoscopeEngine.logic_weights = {k: v / total_weight for k, v in KaleidoscopeEngine.logic_weights.items()}

            insights.append({
                "node_id": dump["node_id"],
                "weighted_logic": dict(KaleidoscopeEngine.logic_weights),
                "insight": f"Processed traits {dump['traits']}"
            })
        logging.info("Data dumps processed.")
        return insights

