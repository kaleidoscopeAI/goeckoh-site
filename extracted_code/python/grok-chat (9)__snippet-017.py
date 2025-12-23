    20  } from '../src/services/api';
       ⋮
    30    const [heart, setHeart] = useState<any>(null);
    31 +  const [validation, setValidation] = useState<any>(null);
    32
       ⋮
    60          setFragments(fr.fragments || []);
    61 +        const v = await getMirrorValidation();
    62 +        setValidation(v.summary || null);
    63        } catch (err) {

