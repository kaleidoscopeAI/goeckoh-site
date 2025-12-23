import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import admin from "firebase-admin";
import serviceAccount from "./firebaseServiceAccount.json";
import { DEFAULT_EMOTIONS } from "../src/services/actuationService";
