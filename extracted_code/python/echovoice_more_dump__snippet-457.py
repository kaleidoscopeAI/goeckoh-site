import { useEffect, useState } from "react";
import { initializeApp } from "firebase/app";
import { getFirestore, doc, onSnapshot } from "firebase/firestore";
export function useActuation(firebaseConfig: any) {
const [projection, setProjection] = useState<{ts:number, n:number, m:number, data:number[]}|null>(null);
const [nodeMods, setNodeMods] = useState<Record<string, any>>({});
