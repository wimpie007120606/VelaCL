import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { chromium } from "playwright";

// Serves an exported site directory under the /VelaCL base path and fails
// unless the rendered page shows real experiment data. Guards against builds
// whose client fetch path misses the GitHub Pages project prefix.
const dir = resolve(process.argv[2] ?? "../../docs");
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".css": "text/css", ".png": "image/png", ".txt": "text/plain" };
const server = createServer(async (req, res) => {
  let path = req.url.split("?")[0];
  if (!path.startsWith("/VelaCL")) { res.writeHead(404); return res.end(); }
  path = path.slice("/VelaCL".length) || "/";
  if (path.endsWith("/")) path += "index.html";
  try {
    const body = await readFile(join(dir, normalize(path)));
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/VelaCL/`, { waitUntil: "networkidle" });
const body = await page.textContent("body");
const methodCards = await page.locator(".cards article").count();
const heatmapCells = await page.locator(".heatmap span").count();
await browser.close();
server.close();
if (body.includes("Experiment data is unavailable") || methodCards < 6 || heatmapCells === 0) {
  console.error(`FAIL: export in ${dir} did not render experiment data (method cards: ${methodCards}, heatmap cells: ${heatmapCells})`);
  process.exit(1);
}
console.log(`OK: export renders experiment data (${methodCards} method cards, ${heatmapCells} heatmap cells)`);
