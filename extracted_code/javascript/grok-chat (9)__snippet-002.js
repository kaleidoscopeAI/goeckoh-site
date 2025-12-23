    24 +export const getHeartSnapshot = async () => {
    25 +  const data = await getSystemStatus();
    26 +  return {
    27 +    gcl: data.gcl,
    28 +    stress: data.stress,
    29 +    mode: data.system_mode,
    30 +    heartSample: data.heart_sample || [],
    31 +    gclHistory: data.gcl_history || [],
    32 +    gclTs: data.gcl_ts || [],
    33 +    emotional: data.emotional_state || {},
    34 +  };
    35 +};
    36 +
    37  const authHeaders = () => {

• Planning to apply the patch using a path relative to the repo root to avoid
  issues with the current working directory.

