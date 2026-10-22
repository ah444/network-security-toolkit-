// Resume after a reload (v5.1 step 1.3c): the server still holds the session's dataset,
// run, running training, export and loaded bundle. GET /api/v1/workspace answers them in
// the shapes their own routes gave, and each module shows its part again.

import { getJSON } from "./api.js";
import { emit } from "./state.js";

export async function resume() {
  try {
    emit("workspace:resumed", await getJSON("/api/v1/workspace"));
  } catch (error) {  // e.g. the server was restarted: start from an empty page
    console.error("Could not resume the workspace:", error);
  }
  emit("steps:restore");
}
