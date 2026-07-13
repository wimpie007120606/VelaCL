import type { NextConfig } from "next";

const pages = process.env.GITHUB_PAGES === "1";

const config: NextConfig = {
  turbopack: { root: process.cwd() },
  output: pages ? "export" : undefined,
  basePath: pages ? "/VelaCL" : undefined,
};

export default config;
