17    stopMirror,
18 -  getHeartSnapshot,
18  } from '../src/services/api';
   ⋮
34          setSystemStatus(status);
36 -        const heartSnap = await getHeartSnapshot();
37 -        setHeart(heartSnap);
35 +        setHeart({
36 +          gcl: status.gcl,
37 +          stress: status.stress,
38 +          mode: status.system_mode,
39 +          heartSample: status.heart_sample || [],
40 +          gclHistory: status.gcl_history || [],
41 +          gclTs: status.gcl_ts || [],
42 +          emotional: status.emotional_state || {},
43 +        });
44        } catch (err) {

