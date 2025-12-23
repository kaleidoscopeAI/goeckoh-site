import fs from "fs";
import path from "path";
import yargs from "yargs";
import { ProjectionService } from "../src/services/projectionService";
import { ConstructLearner } from "../src/services/constructLearner";
const argv = yargs(process.argv.slice(2))
