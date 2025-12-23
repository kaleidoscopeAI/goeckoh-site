assistant: remake this adding all the voice mimicry aspect of the system import React, { useState, useEffect, useCallback, useRef } from 'react';
const J=1.0, K=0.5, L0=1.0, GAMMA=0.2, LAMBDA=0.001, PHI_THRESH=0.55;
const PHRASES = ['hello','water','thank you','help','good morning','yes','no'];
const CONCEPTS = ['calm','stress','joy','focus','flow','anxiety','safety','growth'];
const hammingSim = (a,b) => { let x=(a^b)>>>0,c=0; while(x){c+=x&1;x>>>=1;} return 1-c/32; };
const manhattanDist = (a,b) => Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1])+Math.abs(a[2]-b[2]);
const randU32 = () => (Math.random()*0xFFFFFFFF)>>>0;
const randPos = () => [Math.random()*8-4, Math.random()*8-4, Math.random()*8-4];
const popcount = (n) => { let c=0; for(let i=0;i<32;i++) c+=(n>>>i)&1; return c; };
const clamp = (v,min,max) => Math.max(min,Math.min(max,v));
const stringSim = (a,b) => {
