import { initializeApp } from "firebase/app";
import { getFirestore, doc, setDoc } from "firebase/firestore";
import { SparseMatrix, mul as sparseMul, buildLaplacian } from "./sparse";
import {
