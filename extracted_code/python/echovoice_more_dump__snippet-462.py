import React from "react";
import EmotionalDial from "./components/EmotionalDial";
import { useEmotionalActuation } from "./hooks/useEmotionalActuation";
import { DEFAULT_EMOTIONS } from "./services/actuationService";
import { EmotionLoop } from "./core/emotionLoop";
export default function App() {
