// The unlock page (/unlock, v5.2 step 2.4f), loaded as an ES module: another computer types
// the access token once; the server keeps this session unlocked until the toolkit stops or a
// new token is made on its computer (HS1), and the page goes on to the home page. The token is sent in the request's body, never in
// the address.

import { send } from "./api.js";
import { byId, message } from "./dom.js";

async function unlock(event) {
  event.preventDefault();
  const status = byId("unlockStatus");
  byId("unlockButton").disabled = true;
  try {
    const answer = await send("/api/v1/unlock", "POST", { token: byId("accessToken").value });
    message(status, "success", answer.message);
    window.location.assign("/");
  } catch (error) {
    message(status, "error", "Error: " + error.message);
    byId("unlockButton").disabled = false;
    byId("accessToken").select();
  }
}

byId("unlockForm").addEventListener("submit", unlock);
