import { initializeApp } from "firebase/app";
import { getFirestore, doc, setDoc, onSnapshot } from "firebase/firestore";
import { SparseMatrix } from "./sparse"; // your sparse matutils
// NOTE: implement or import sparse mat utilities (mul, add, diag, laplacian)
