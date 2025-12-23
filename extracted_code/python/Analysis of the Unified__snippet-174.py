    if extra:
        prompt += "\n" + extra
    try:
        import requests
        resp = requests.post(f"{self.base_url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False, "options":{"temperature":0.0}}, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        txt = resp.json().get("response","")
        # find first {...}
        s = txt.find("{"); e = txt.rfind("}")
        if s==-1 or e==-1:
            return None
        js = txt[s:e+1]
        obj = json.loads(js)
        # clamp
        obj2 = {}
        if 'mix_delta' in obj:
            obj2['mix_delta'] = float(np.clip(float(obj['mix_delta']), -0.2, 0.2))
        if 'gamma_delta' in obj:
            obj2['gamma_delta'] = float(np.clip(float(obj['gamma_delta']), -0.5, 0.5))
        if 'bond_scale' in obj:
            obj2['bond_scale'] = float(np.clip(float(obj['bond_scale']), 0.5, 1.5))
        return obj2
    except Exception as e:
        logger.warning("LLMAdapter failed: %s", e)
        return None

