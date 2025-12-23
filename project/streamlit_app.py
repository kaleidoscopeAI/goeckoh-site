import json, time, pathlib, streamlit as st
st.set_page_config(page_title="Organic AI – Seed Monitor", layout="wide")
st.title("🌱 Organic AI – Seed Monitor")
state_path=pathlib.Path('/mnt/data/seed_state.json')
while True:
    if state_path.exists():
        data=json.loads(state_path.read_text())
        c1,c2=st.columns(2)
        with c1:
            st.metric("Energy", data.get('energy')); st.metric("Complexity", data.get('complexity')); st.json(data.get('traits'))
        with c2:
            st.write("DNA", data.get('dna')); st.write("Anneal Temp", data.get('anneal_temp')); st.json(data.get('gears'))
    else:
        st.info("Waiting for seed_state.json ... (run main.py)")
    time.sleep(1.0)
