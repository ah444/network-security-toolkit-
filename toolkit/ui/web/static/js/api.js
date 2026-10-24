// The only place the page talks to the server (/api/v1, docs/web_api.md). Every POST and
// DELETE carries the session's CSRF token from the page's meta tag (S10); without it the
// server answers 403 CSRF_TOKEN. An error answer becomes an Error with the server's
// message, so every caller shows errors the same way.

function csrfToken() {
  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta ? meta.content : "";
}

async function answer(response) {
  const data = await response.json().catch(() => null);
  if (!response.ok || data === null) {
    throw new Error(data && data.error ? data.error.message : "HTTP " + response.status);
  }
  return data;
}

// GET with optional query parameters
export async function getJSON(url, params) {
  const query = params ? "?" + new URLSearchParams(params) : "";
  return answer(await fetch(url + query));
}

// POST or DELETE: a FormData body is sent as it is (file uploads), anything else as JSON
export async function send(url, method, body) {
  const headers = { "X-CSRF-Token": csrfToken() };
  let payload = body;
  if (body !== undefined && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  return answer(await fetch(url, { method: method, headers: headers, body: payload }));
}
