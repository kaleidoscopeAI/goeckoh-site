import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js";

const container = document.getElementById('container');
const statusEl = document.getElementById('status');
const wsurlEl = document.getElementById('wsurl');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0c10);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 3.2);

const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Points geometry (allocated on first frame)
let points = null;
let positions = null; // Float32Array
const geom = new THREE.BufferGeometry();
const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true, opacity: 0.95, transparent: true });

// Color buffer (dynamic: hot colors based on radial distance)
let colors = null;

function ensureCapacity(npoints) {

