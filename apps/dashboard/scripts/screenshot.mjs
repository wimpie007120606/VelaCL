import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
await page.goto("http://localhost:3000",{waitUntil:"networkidle"});
await mkdir("../../reports/assets",{recursive:true});
await page.screenshot({path:"../../reports/assets/dashboard.png",fullPage:true});
await browser.close();

