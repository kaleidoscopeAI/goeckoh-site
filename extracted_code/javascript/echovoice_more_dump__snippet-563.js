const firebaseConfig = process.env.REACT_APP_FIREBASE_CONFIG ? JSON.parse(process.env.REACT_APP_FIREBASE_CONFIG) : undefined;
const { e, setE, setEmotion, resetNeutral } = useEmotionalActuation(firebaseConfig);
