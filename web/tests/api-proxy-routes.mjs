import assert from "node:assert/strict";

const baseUrl = process.env.TEST_BASE_URL || "http://127.0.0.1:3000";

async function request(path, init) {
  const response = await fetch(new URL(path, baseUrl), init);
  assert.notEqual(response.status, 404, `${path} should be handled by the web app`);
  return response;
}

await request("/models");
await request("/history?limit=10&offset=0");
await request("/predict", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ smiles: "CCO" }),
});

console.log("API proxy routes are present.");
