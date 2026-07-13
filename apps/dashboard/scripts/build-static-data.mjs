import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../../..");
const runs = resolve(root, "experiments/runs");
const methods = [
  "static",
  "naive",
  "random_replay",
  "active_balanced_replay",
  "active_uncertainty_replay",
  "active_diversity_replay",
];
const summary = JSON.parse(await readFile(resolve(runs, "summary.json"), "utf8"));
const experiments = {};
for (const method of methods) {
  experiments[method] = JSON.parse(
    await readFile(resolve(runs, method, "metrics.json"), "utf8"),
  );
}
const output = resolve(root, "apps/dashboard/public");
await mkdir(output, { recursive: true });
await writeFile(
  resolve(output, "experiments.json"),
  JSON.stringify({ summary, experiments }),
);
