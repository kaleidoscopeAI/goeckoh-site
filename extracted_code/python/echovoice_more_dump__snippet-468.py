import { useEffect, useState } from "react";
import { initializeApp } from "firebase/app";
import { getFirestore, doc, onSnapshot } from "firebase/firestore";
export function useActuation(firebaseConfig: any, nodeCount = 4) {
const [projection, setProjection] = useState<any|null>(null);
const [nodeMods, setNodeMods] = useState<Record<string, any>>({});
