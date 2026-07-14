import type { NextConfig } from "next";

const pages = process.env.GITHUB_PAGES === "1";

const config: NextConfig = {
  turbopack: { root: process.cwd() },
  output: pages ? "export" : undefined,
  basePath: pages ? "/VelaCL" : undefined,
  // The client fetches `${NEXT_PUBLIC_BASE_PATH}/experiments.json`; derive it
  // from the same flag as basePath so a build can't ship a mismatched pair.
  env: { NEXT_PUBLIC_BASE_PATH: pages ? "/VelaCL" : "" },
};

export default config;
