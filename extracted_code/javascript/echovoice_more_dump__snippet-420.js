const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const projRef = doc(db, "actuation", "latestProjection");
const unsubProj = onSnapshot(projRef, (snap) => {
