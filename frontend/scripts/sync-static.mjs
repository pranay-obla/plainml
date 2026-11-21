// Copies the built site (out/) into the Python package, where `plainml web` serves it from and
// where it ships in the wheel. Run by `npm run build`.
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "out");
const target = join(here, "..", "..", "plainml", "web", "static");

if (!existsSync(join(out, "index.html"))) {
  console.error("No out/index.html: run `next build` first.");
  process.exit(1);
}
rmSync(target, { recursive: true, force: true });
cpSync(out, target, { recursive: true });
console.log(`Copied the site into ${target}`);
