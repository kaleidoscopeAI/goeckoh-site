import React, { useEffect } from "react";
import { ActuationService } from "./services/actuationService";
import { makeSimpleGraph } from "./utils/graphHelpers"; // implement helper or inline
const firebaseConfig = null; // or your firebase config
const graph = makeSimpleGraph(4); // helper that returns SparseMatrix
const cfg = {
