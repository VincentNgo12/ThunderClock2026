import { defineConfig } from 'vite';

// Port 5180 so the pitch can run next to the spine frontend (5173). Everything is served locally.
// /backend/* is proxied to the telemetry backend's HTTP API (Connect vest button), so the browser never
// needs CORS for it. Override the target with BACKEND_HTTP=http://host:port when starting vite.
const backend = process.env.BACKEND_HTTP || 'http://127.0.0.1:8000';
const proxy = { '/backend': { target: backend, changeOrigin: true, rewrite: (p) => p.replace(/^\/backend/, '') } };

export default defineConfig({
  server: { port: 5180, host: '127.0.0.1', proxy },
  preview: { port: 5181, host: '127.0.0.1', proxy },
  build: { chunkSizeWarningLimit: 1024 },
});
