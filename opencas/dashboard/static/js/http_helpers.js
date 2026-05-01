(function (global) {
  function normalizeRelativeUrl(url) {
    if (typeof url !== 'string') return url;
    if (url.startsWith('//')) return url;
    if (/^(?:[a-z]+:|data:|blob:|mailto:|tel:|ws:|wss:)/i.test(url)) return url;
    return url.startsWith('/') ? url.slice(1) : url;
  }

  if (!global.__openCASUrlNormalizerInstalled) {
    global.__openCASUrlNormalizerInstalled = true;
    const rawFetch = global.fetch?.bind(global);
    if (rawFetch) {
      global.fetch = (input, init) => {
        if (typeof input === 'string') {
          return rawFetch(normalizeRelativeUrl(input), init);
        }
        if (input && typeof input.url === 'string' && input.url.startsWith('/')) {
          const request = new Request(normalizeRelativeUrl(input.url), input);
          return rawFetch(request, init);
        }
        return rawFetch(input, init);
      };
    }

    const rawOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function (method, url, ...rest) {
      return rawOpen.call(this, method, normalizeRelativeUrl(url), ...rest);
    };
  }

  function buildUrl(base, params) {
    const query = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      if (typeof value === 'string' && !value.trim()) return;
      query.set(key, String(value));
    });
    const suffix = query.toString();
    return suffix ? `${base}?${suffix}` : base;
  }

  function resolveUrl(url) {
    return normalizeRelativeUrl(url);
  }

  async function safeJson(response, fallback = null) {
    try {
      return await response.json();
    } catch (_error) {
      return fallback;
    }
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`${options?.method || 'GET'} ${url} failed: ${response.status}`);
    }
    return await safeJson(response, {});
  }

  async function fetchJsonOrNull(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) return null;
    return await safeJson(response, null);
  }

  async function fetchJsonAllSettled(requests) {
    const settled = await Promise.allSettled(
      (requests || []).map(async (request) => {
        const spec = typeof request === 'string' ? { url: request } : (request || {});
        const response = await fetch(spec.url, spec.options);
        if (!response.ok) return null;
        return await safeJson(response, null);
      })
    );
    return settled.map((item) => (item.status === 'fulfilled' ? item.value : null));
  }

  global.OpenCASHttp = {
    buildUrl,
    fetchJson,
    fetchJsonOrNull,
    fetchJsonAllSettled,
    resolveUrl,
    safeJson,
  };
})(window);
