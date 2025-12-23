import React, { useEffect, useState, useRef } from "react";
import Constructs3D from "../components/Constructs3D";
import ActuationHeatmap from "../components/ActuationHeatmap";
import { useEmotionalActuation } from "../hooks/useEmotionalActuation";
import { ProjectionService } from "../services/projectionService";
import Chart from "chart.js/auto";
import axios from "axios";
