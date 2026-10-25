import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PUBLIC_DIR = fileURLToPath(new URL("./public/", import.meta.url));
const PORT = Number(process.env.PORT) || 3000;

const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".bin": "application/octet-stream"
};

function send(response, status, body, headers = {}) {
  response.writeHead(status, {
    "Content-Type": "text/plain; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
    ...headers
  });
  response.end(body);
}

function resolveFile(pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }

  const target = resolve(PUBLIC_DIR, `.${decoded}`);
  if (target !== PUBLIC_DIR.slice(0, -1) && !target.startsWith(PUBLIC_DIR)) {
    return null;
  }

  try {
    const stats = statSync(target);
    if (stats.isDirectory()) {
      return { path: resolve(target, "index.html"), stats: statSync(resolve(target, "index.html")) };
    }
    return { path: target, stats };
  } catch {
    return null;
  }
}

const server = createServer((request, response) => {
  if (request.method !== "GET" && request.method !== "HEAD") {
    send(response, 405, "Method not allowed", { Allow: "GET, HEAD" });
    return;
  }

  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
  const file = resolveFile(url.pathname);

  if (!file) {
    send(response, 404, "Not found");
    return;
  }

  const extension = extname(file.path).toLowerCase();
  const headers = {
    "Content-Type": MIME_TYPES[extension] ?? "application/octet-stream",
    "Content-Length": file.stats.size,
    "Cache-Control": extension === ".html" ? "no-cache" : "public, max-age=0, must-revalidate"
  };

  response.writeHead(200, headers);
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  createReadStream(file.path).pipe(response);
});

server.listen(PORT, () => {
  console.log(`EmojiBot running at http://localhost:${PORT}`);
});
