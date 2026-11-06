/* PixelForge service worker: app shell cached, HTML is network-first so
 * deploys show up immediately (no hard-reload needed).
 * API/uploads/results are never intercepted (always network). */
const CACHE = 'pixelforge-v4';
const SHELL = ['/', '/manifest.webmanifest', '/icons/icon-192.png', '/icons/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // Never cache API, uploads or results — always network.
  if (url.pathname.startsWith('/api/')) return;
  if (e.request.method !== 'GET') return;
  // Only handle same-origin requests; let CDNs (tailwind/lucide/fonts) use browser cache.
  if (url.origin !== self.location.origin) return;

  const isHTML =
    e.request.mode === 'navigate' ||
    url.pathname === '/' ||
    url.pathname.endsWith('.html') ||
    url.pathname === '/manifest.webmanifest' ||
    url.pathname === '/sw.js';

  if (isHTML) {
    // Network-first: always try fresh HTML/shell, fall back to cache offline.
    e.respondWith(
      fetch(e.request, { cache: 'no-store' }).then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      }).catch(() => caches.match(e.request).then((hit) => hit || caches.match('/')))
    );
    return;
  }

  // Static assets (icons, etc.): stale-while-revalidate.
  e.respondWith(
    caches.match(e.request).then((hit) => {
      const network = fetch(e.request).then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      }).catch(() => hit);
      return hit || network;
    })
  );
});
