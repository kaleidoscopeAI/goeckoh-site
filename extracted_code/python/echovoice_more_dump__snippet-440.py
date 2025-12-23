import { ActuationService } from "./services/actuationService";
import { initializeApp, getFirestore } from "firebase-admin/firestore";
const firebaseApp = initializeApp();
const firestore = getFirestore(firebaseApp);
const actuationService = new ActuationService(firestore);
