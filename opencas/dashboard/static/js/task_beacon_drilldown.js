(function (root, factory) {
  const renderTaskBeaconSummary = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      renderTaskBeaconSummary,
      taskBeaconBucketSignature: renderTaskBeaconSummary.taskBeaconBucketSignature,
      mergeTaskBeaconSummary: renderTaskBeaconSummary.mergeTaskBeaconSummary,
    };
  }
  root.renderTaskBeaconSummary = renderTaskBeaconSummary;
  root.taskBeaconBucketSignature = renderTaskBeaconSummary.taskBeaconBucketSignature;
  root.mergeTaskBeaconSummary = renderTaskBeaconSummary.mergeTaskBeaconSummary;
})(typeof globalThis !== 'undefined' ? globalThis : window, function () {
  function _safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function _escape(value, escapeHtml) {
    if (typeof escapeHtml === 'function') {
      return escapeHtml(value);
    }
    return String(value ?? '');
  }

  function _bucketData(taskBeacon) {
    const viewModelBuckets = _safeArray(taskBeacon?.view_model?.buckets);
    const fallback = ['now', 'next', 'later'].map((state) => ({
      state,
      count: 0,
      item: null,
      items: [],
      reason: 'No matching build/test fragments.',
    }));
    return (viewModelBuckets.length ? viewModelBuckets : fallback).map((bucket) => ({
      state: bucket?.state || 'now',
      count: Number(bucket?.count ?? 0),
      item: bucket?.item || null,
      items: _safeArray(bucket?.items),
      reason: String(bucket?.reason || 'No matching build/test fragments.'),
    }));
  }

  function _bucketItemLabel(item) {
    if (!item) return 'No matching fragments.';
    return item.title || item.task_id || 'Untitled fragment';
  }

  function _bucketItemMeta(item, escapeHtml) {
    const parts = [];
    if (item?.task_id) parts.push(`<span>${_escape(item.task_id, escapeHtml)}</span>`);
    if (item?.owner) parts.push(`<span>${_escape(item.owner, escapeHtml)}</span>`);
    if (item?.section) parts.push(`<span>${_escape(item.section, escapeHtml)}</span>`);
    if (item?.merged_count && Number(item.merged_count) > 1) parts.push(`<span>${_escape(`${item.merged_count} merged`, escapeHtml)}</span>`);
    return parts.join('');
  }

  function _bucketItemExcerpt(item, escapeHtml) {
    if (!item || !item.excerpt) return '';
    return `<div class="muted">${_escape(item.excerpt, escapeHtml)}</div>`;
  }

  function _detailItems(taskBeacon, bucket) {
    const detailsByState = taskBeacon?.details && typeof taskBeacon.details === 'object' ? taskBeacon.details : {};
    const detailItems = _safeArray(detailsByState?.[bucket.state]);
    return detailItems.length ? detailItems : _safeArray(bucket.items);
  }

  function _renderFragmentList(taskBeacon, bucket, escapeHtml) {
    const fragments = _detailItems(taskBeacon, bucket);
    if (!fragments.length) {
      return '<p class="muted">No merged fragments to show.</p>';
    }
    let html = '<div class="task-beacon-fragment-list">';
    fragments.forEach((fragment) => {
      html += '<article class="task-beacon-fragment">';
      html += '<div class="task-beacon-bucket-item-head">';
      html += `<span class="badge">${_escape(fragment?.status || 'unknown', escapeHtml)}</span>`;
      html += `<span class="task-beacon-fragment-title">${_escape(_bucketItemLabel(fragment), escapeHtml)}</span>`;
      html += '</div>';
      html += `<div class="task-beacon-bucket-item-meta muted">${_bucketItemMeta(fragment, escapeHtml)}</div>`;
      html += _bucketItemExcerpt(fragment, escapeHtml);
      html += '</article>';
    });
    html += '</div>';
    return html;
  }

  function _renderBucketSummary(bucket, escapeHtml) {
    const countLabel = `${bucket.count} item${bucket.count === 1 ? '' : 's'}`;
    let html = '<summary class="task-beacon-bucket-summary">';
    html += '<span class="task-beacon-bucket-summary-line">';
    html += `<span class="badge">${_escape(bucket.state, escapeHtml)}</span>`;
    html += `<span class="task-beacon-bucket-count">${countLabel}</span>`;
    html += `<span class="task-beacon-bucket-reason muted">${_escape(bucket.reason, escapeHtml)}</span>`;
    html += '</span>';
    html += '</summary>';
    return html;
  }

  function _renderBucketDetails(taskBeacon, bucket, escapeHtml) {
    let html = '<div class="task-beacon-bucket-details">';
    html += _renderFragmentList(taskBeacon, bucket, escapeHtml);
    html += '</div>';
    return html;
  }

  function taskBeaconBucketSignature(taskBeacon) {
    if (typeof taskBeacon?.bucket_signature === 'string' && taskBeacon.bucket_signature) {
      return taskBeacon.bucket_signature;
    }
    return _bucketData(taskBeacon)
      .map((bucket) => `${bucket.state}:${bucket.count}`)
      .join('|');
  }

  function mergeTaskBeaconSummary(current, incoming) {
    if (!current) return incoming || current;
    if (!incoming) return current;
    return taskBeaconBucketSignature(current) === taskBeaconBucketSignature(incoming) ? current : incoming;
  }

  function renderTaskBeaconSummary(taskBeacon, escapeHtml) {
    let html = '<h4>Task Beacon</h4>';
    if (!taskBeacon || taskBeacon.error || taskBeacon.available === false) {
      const message = taskBeacon?.error
        ? `Task beacon unavailable: ${_escape(taskBeacon.error, escapeHtml)}`
        : 'No matching build/test fragments.';
      return `${html}<p class="muted">${message}</p>`;
    }

    const buckets = _bucketData(taskBeacon || {});
    html += '<div class="chat-task-list task-beacon-buckets mt-3">';
    for (const bucket of buckets) {
      html += `<details class="chat-task-item task-beacon-bucket" data-state="${_escape(bucket.state, escapeHtml)}">`;
      html += _renderBucketSummary(bucket, escapeHtml);
      html += _renderBucketDetails(taskBeacon || {}, bucket, escapeHtml);
      html += '</details>';
    }
    html += '</div>';
    return html;
  }

  renderTaskBeaconSummary.taskBeaconBucketSignature = taskBeaconBucketSignature;
  renderTaskBeaconSummary.mergeTaskBeaconSummary = mergeTaskBeaconSummary;
  return renderTaskBeaconSummary;
});
