import { useState, useEffect } from "react";
import { actuationService, EVector } from "../services/actuationService";
import axios from "axios";
export function useEmotionalActuation() {
const [e, setEState] = useState<EVector>({ ...actuationService.e });
const setE = (patch: Partial<EVector>) => {
