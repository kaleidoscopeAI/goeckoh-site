import { NodeState, Edge, Vec3 } from "./types";
import { randomUUID } from "crypto";

function add(a: Vec3, b: Vec3): Vec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
function len(a: Vec3): number { return Math.sqrt(a[0]**2 + a[1]**2 + a[2]**2); }
function normalize(a: Vec3): Vec3 { const l = len(a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }
function gaussian(): number { return Math.random() * 2 - 1; } // Approx N(0,1)
function clip(v: number, min: number, max: number): number { return Math.max(min, Math.min(max, v)); }

