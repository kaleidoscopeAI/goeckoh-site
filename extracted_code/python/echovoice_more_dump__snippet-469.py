import { ActuationService } from "../src/services/actuationService";
import { SparseMatrix } from "../src/services/sparse";
const makeSimpleGraph = (n: number): SparseMatrix => {
const rows = new Array(n);
for (let i = 0; i < n; i++) {
const neighbors: number[] = [];
const weights: number[] = [];
