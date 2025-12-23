import * as math from "mathjs";
import { jacobian } from "./jacobian";
import { Node, EmotionalVector } from "../types";
export function spectralRadius(node: Node, e: EmotionalVector): number {
const J = jacobian(node, e);
const eig = math.eigs(J);
