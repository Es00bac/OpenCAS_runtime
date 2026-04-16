(function (global) {
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
    safeJson,
  };
})(window);
