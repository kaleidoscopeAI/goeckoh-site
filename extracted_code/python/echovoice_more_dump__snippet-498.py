import React from "react";
import { useEmotionalActuation } from "../hooks/useEmotionalActuation";
import EmotionalDial from "./EmotionalDial";
import { DEFAULT_EMOTIONS } from "../services/actuationService";
export default function ActuationTestHarness() {
const { e, setE, resetNeutral } = useEmotionalActuation();
