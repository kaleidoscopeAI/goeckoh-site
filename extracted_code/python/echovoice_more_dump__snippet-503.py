import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import fs from "fs";
import path from "path";
import admin from "firebase-admin";
import serviceAccount from "./firebaseServiceAccount.json";
const app = express();
