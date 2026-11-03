// Web trainer page (/dynamic_ids), loaded as an ES module: module scripts run once the
// page is parsed, so every element exists when the modules wire up their handlers.
// The modules talk to each other through events (see state.js), not by calling each other.
// Listeners run in the order the modules are set up: data before train (a resumed dataset
// comes before its run), and steps last; then the page resumes from the server.

import { setup as setupData } from "./data.js";
import { setup as setupExplain } from "./explain.js";
import { setup as setupExplore } from "./explore.js";
import { setup as setupPredict } from "./predict.js";
import { resume } from "./resume.js";
import { setup as setupSteps } from "./steps.js";
import { setup as setupTrain } from "./train.js";

setupData();
setupExplore();
setupTrain();
setupExplain();
setupPredict();
setupSteps();
resume();
