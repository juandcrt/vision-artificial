const fs = require("node:fs");
const path = require("node:path");

const apiUrl = process.env.VF_API_URL?.trim() || "";

if (apiUrl && !/^https?:\/\//i.test(apiUrl)) {
  throw new Error("VF_API_URL debe comenzar con http:// o https://.");
}

const output = path.resolve("dist");
fs.mkdirSync(output, { recursive: true });
fs.cpSync(path.resolve("frontend"), output, { recursive: true });
fs.writeFileSync(
  path.join(output, "config.js"),
  `window.VF_API_URL = ${JSON.stringify(apiUrl)};\n`,
);
