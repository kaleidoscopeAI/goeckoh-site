17    stopMirror,
18 +  getHeartSnapshot,
19  } from '../src/services/api';
   ⋮
28    const [selecting, setSelecting] = useState(false);
29 +  const [heart, setHeart] = useState<any>(null);
30
   ⋮
35          setSystemStatus(status);
36 +        const heartSnap = await getHeartSnapshot();
37 +        setHeart(heartSnap);
38        } catch (err) {
   ⋮
74
75 +  const heartSample = heart?.heartSample || [];
76 +  const heartMode = heart?.mode;
77 +  const heartStress = heart?.stress;
78 +  const heartEmo = heart?.emotional || {};
79 +
80    // Fallback data for chart if not provided by backend

