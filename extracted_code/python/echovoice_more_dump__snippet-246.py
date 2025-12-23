def speculate(snapshot: dict):
    if 'sensors' in snapshot:
        adc = np.array(snapshot['adc_raw'])  # Mock sensor data
        grad = sensor_grad(1.0, adc)
        return {"sensor_update": grad.tolist()}
    # ... Previous

frontend/src/App.tsx (Viz hardware state: overlay CPU/brightness from WS snapshot)
