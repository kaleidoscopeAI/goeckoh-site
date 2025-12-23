import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import admin from "firebase-admin";
import serviceAccount from "./firebaseServiceAccount.json";
import { DEFAULT_EMOTIONS, EVector } from "../src/services/actuationService";
