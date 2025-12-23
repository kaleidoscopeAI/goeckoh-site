import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import fs from "fs";
import path from "path";
import admin from "firebase-admin";
import serviceAccount from "./firebaseServiceAccount.json";
const app = express();
