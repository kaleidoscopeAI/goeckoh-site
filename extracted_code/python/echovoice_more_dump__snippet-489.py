const DEFAULT_DOC_PATH = "actuation/current";
const DEFAULT_HISTORY_COLLECTION = "actuation/history";
const DEFAULT_NORMALIZE: ActuationDoc["normalized"] = "none";
const CLAMP_MIN = -1.0;
const CLAMP_MAX = 1.0;
const DEFAULT_SMOOTHING_MS = 300; // linear ramp time for local smoothing (ms)
const RATE_LIMIT_MS = 200; // min interval between writes from same client (ms)
