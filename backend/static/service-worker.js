const CACHE_NAME = 'agri-detect-v3';
const PRECACHE_URLS = [
  '/',
  '/static/style.css',
  '/static/app.js',
  '/static/manifest.json?v=3',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.map((key) => {
        if (key !== CACHE_NAME) {
          return caches.delete(key);
        }
        return Promise.resolve();
      })
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Let POST requests and API calls pass through to network (don't cache)
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.pathname.startsWith('/detect') || url.pathname.startsWith('/api')) {
    return event.respondWith(fetch(event.request).catch(() => new Response(null, { status: 503 })));
  }

  // For navigation requests, try network first, fallback to cache
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).then((resp) => {
        return resp;
      }).catch(() => caches.match('/'))
    );
    return;
  }

  // For other GET requests, respond cache-first then fetch-and-update
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        return caches.open(CACHE_NAME).then((cache) => {
          // Avoid caching opaque cross-origin requests
          try { cache.put(event.request, response.clone()); } catch (e) {}
          return response;
        });
      }).catch(() => {
        return caches.match('/static/icons/icon-192.png');
      });
    })
  );
});
