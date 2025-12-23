69
70 +export const getMirrorValidation = async () => {
71 +  const res = await fetch(`${API_BASE_URL}/mirror/validate`, { headers:
     authHeaders() });
72 +  if (!res.ok) throw new Error(`mirror/validate ${res.status}`);
73 +  return res.json();
74 +};
75 +
76 +export const resetMirrorMetrics = async () => {
77 +  const res = await fetch(`${API_BASE_URL}/mirror/reset_metrics`, {
78 +    method: 'POST',
79 +    headers: { ...authHeaders() },
80 +  });
81 +  if (!res.ok) throw new Error(`mirror/reset_metrics ${res.status}`);
82 +  return res.json();
83 +};
84 +
85  export const startMirror = async (payload: Record<string, unknown>) =>
    {

