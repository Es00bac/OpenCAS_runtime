(function (global) {
  "use strict";

  function _hashColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash * 31 + str.charCodeAt(i)) & 0xffffffff;
    }
    return `hsl(${Math.abs(hash) % 360} 70% 58%)`;
  }

  function defaultRetrievalWeights() {
    return {
      semantic_score: 0.30,
      keyword_score: 0.20,
      recency_score: 0.15,
      salience_score: 0.10,
      graph_score: 0.10,
      emotional_resonance: 0.08,
      temporal_echo: 0.04,
      reliability: 0.03,
    };
  }

  function renderMemoryStats(d) {
    return `<div class="stat-grid">
      <div><div class="stat-value">${d.episode_count}</div><div class="stat-label">Episodes</div></div>
      <div><div class="stat-value">${d.memory_count}</div><div class="stat-label">Memories</div></div>
      <div><div class="stat-value">${d.edge_count}</div><div class="stat-label">Edges</div></div>
      <div><div class="stat-value">${d.comacted_count || d.compacted_count}</div><div class="stat-label">Compacted</div></div>
    </div>`;
  }

  function memoryApp() {
    return {
      landscape: { stats: {}, nodes: [], edges: [], projection: { groups: [] } },
      landscapeLoading: false,
      landscapeChart: null,
      query: '',
      sessionId: '',
      kind: '',
      emotion: '',
      maxAgeDays: '',
      limit: 140,
      minEdgeConfidence: 0.18,
      edgeKind: '',
      projectionMethod: 'auto',
      includeMemories: 'true',
      showEdges: 'true',
      colorBy: 'emotion',
      viewMode: 'atlas',
      sizeBy: 'salience',
      neighborhoodHops: 1,
      activeLowerPanel: 'timeline',
      memoryValue: null,
      memoryValueLoading: false,
      hiddenKinds: [],
      hiddenEmotions: [],
      selectedNodeId: '',
      nodeDetail: null,
      nodeDetailLoading: false,
      retrieval: null,
      retrievalLoading: false,
      retrievalQuery: '',
      retrievalLimit: 12,
      retrievalMinConfidence: 0.15,
      retrievalLambda: 0.5,
      retrievalExpandGraph: 'true',
      showRetrievalOnAtlas: false,
      weights: defaultRetrievalWeights(),
      somatic: null,
      musubiState: null,
      somaticLoading: false,
      musubiLoading: false,
      globalStats: null,
      initMemory() {
        global.__openCASMemoryApp = this;
        this.loadGlobalStats();
        this.loadMemoryValue();
        this.loadLandscape();
        this.loadSomaticState();
        this.loadMusubiState();
      },
      async loadGlobalStats() {
        try {
          const response = await fetch('/api/memory/stats');
          if (response.ok) this.globalStats = await response.json();
        } catch (e) { console.error(e); }
      },
      async loadLandscape() {
        this.landscapeLoading = true;
        try {
          const params = new URLSearchParams({
            limit: String(this.limit),
            min_edge_confidence: String(this.minEdgeConfidence),
            include_memories: String(String(this.includeMemories) === 'true'),
            method: this.projectionMethod,
          });
          if (this.query.trim()) params.set('query', this.query.trim());
          if (this.sessionId.trim()) params.set('session_id', this.sessionId.trim());
          if (this.kind) params.set('kind', this.kind);
          if (this.emotion) params.set('emotion', this.emotion);
          if (String(this.maxAgeDays).trim()) params.set('max_age_days', String(this.maxAgeDays).trim());
          if (this.edgeKind) params.set('edge_kind', this.edgeKind);
          const r = await fetch('/api/memory/landscape?' + params.toString());
          const d = await r.json();
          this.landscape = d || { stats: {}, nodes: [], edges: [], projection: { groups: [] } };
          const selectedStillVisible = (this.landscape.nodes || []).some(node => node.node_id === this.selectedNodeId);
          if (!selectedStillVisible) {
            this.selectedNodeId = this.landscape.nodes?.[0]?.node_id || '';
          }
          await this.loadNodeDetail();
          this.renderLandscapeChart();
        } catch (e) {
          console.error(e);
        }
        this.landscapeLoading = false;
      },
      resetLandscapeFilters() {
        this.query = '';
        this.sessionId = '';
        this.kind = '';
        this.emotion = '';
        this.maxAgeDays = '';
        this.limit = 140;
        this.minEdgeConfidence = 0.18;
        this.edgeKind = '';
        this.projectionMethod = 'auto';
        this.includeMemories = 'true';
        this.showEdges = 'true';
        this.colorBy = 'emotion';
        this.viewMode = 'atlas';
        this.sizeBy = 'salience';
        this.neighborhoodHops = 1;
        this.hiddenKinds = [];
        this.hiddenEmotions = [];
        this.nodeDetail = null;
        this.loadLandscape();
      },
      async setLowerPanel(panel) {
        this.activeLowerPanel = panel;
        if (panel === 'value') {
          await this.loadMemoryValue();
        } else if (panel === 'somatic') {
          await Promise.all([this.loadSomaticState(), this.loadMusubiState()]);
        }
      },
      async loadSomaticState() {
        this.somaticLoading = true;
        try {
          const response = await fetch('/api/identity/somatic');
          if (response.ok) this.somatic = await response.json();
        } catch (e) { console.error(e); }
        this.somaticLoading = false;
      },
      async loadMusubiState() {
        this.musubiLoading = true;
        try {
          const response = await fetch('/api/identity/musubi');
          if (response.ok) this.musubiState = await response.json();
        } catch (e) { console.error(e); }
        this.musubiLoading = false;
      },
      async loadMemoryValue(force = false) {
        if (this.memoryValue && !force) return;
        this.memoryValueLoading = true;
        try {
          const response = await fetch('/api/operations/memory-value');
          if (!response.ok) throw new Error('memory value unavailable');
          this.memoryValue = await response.json();
        } catch (e) {
          console.error(e);
          this.memoryValue = null;
        }
        this.memoryValueLoading = false;
      },
      filterBySession(sessionId) {
        this.sessionId = sessionId;
        this.loadLandscape();
      },
      toggleKindVisibility(kind) {
        if (!kind) return;
        if (this.hiddenKinds.includes(kind)) {
          this.hiddenKinds = this.hiddenKinds.filter(item => item !== kind);
        } else {
          this.hiddenKinds = [...this.hiddenKinds, kind];
        }
        this.renderLandscapeChart();
      },
      toggleEmotionVisibility(emotion) {
        if (!emotion) return;
        if (this.hiddenEmotions.includes(emotion)) {
          this.hiddenEmotions = this.hiddenEmotions.filter(item => item !== emotion);
        } else {
          this.hiddenEmotions = [...this.hiddenEmotions, emotion];
        }
        this.renderLandscapeChart();
      },
      clearLegendFilters() {
        this.hiddenKinds = [];
        this.hiddenEmotions = [];
        this.renderLandscapeChart();
      },
      resetRetrievalWeights() {
        this.weights = defaultRetrievalWeights();
        this.retrievalLimit = 12;
        this.retrievalMinConfidence = 0.15;
        this.retrievalLambda = 0.5;
        this.retrievalExpandGraph = 'true';
      },
      emotionOptions() {
        const visible = this.landscape?.stats?.emotion_distribution || {};
        const global_ = this.globalStats?.affect_distribution || {};
        const merged = { ...global_, ...visible };
        return Object.keys(merged).sort();
      },
      edgeKindOptions() {
        const visible = Object.keys(this.landscape?.stats?.edge_kind_distribution || {});
        const allKinds = ['semantic', 'emotional', 'temporal', 'conceptual', 'relational', 'causal', 'distilled_from'];
        return Array.from(new Set([...allKinds, ...visible])).sort();
      },
      visibleNodes() {
        return (this.landscape.nodes || []).filter(node => typeof node.x === 'number' && typeof node.y === 'number');
      },
      activeNodeSet() {
        const nodes = this.visibleNodes();
        if (this.viewMode !== 'selected' || !this.selectedNodeId) {
          return new Set(nodes.map(node => node.node_id));
        }
        const adjacency = new Map();
        (this.landscape.edges || []).forEach(edge => {
          if (!adjacency.has(edge.source_node_id)) adjacency.set(edge.source_node_id, new Set());
          if (!adjacency.has(edge.target_node_id)) adjacency.set(edge.target_node_id, new Set());
          adjacency.get(edge.source_node_id).add(edge.target_node_id);
          adjacency.get(edge.target_node_id).add(edge.source_node_id);
        });
        const visited = new Set([this.selectedNodeId]);
        let frontier = new Set([this.selectedNodeId]);
        for (let step = 0; step < Number(this.neighborhoodHops || 1); step += 1) {
          const next = new Set();
          frontier.forEach(nodeId => {
            (adjacency.get(nodeId) || new Set()).forEach(otherId => {
              if (!visited.has(otherId)) {
                visited.add(otherId);
                next.add(otherId);
              }
            });
          });
          frontier = next;
          if (!frontier.size) break;
        }
        return visited;
      },
      displayNodes() {
        const active = this.activeNodeSet();
        return this.visibleNodes().filter(node => {
          if (!active.has(node.node_id)) return false;
          if (this.hiddenKinds.includes(node.kind)) return false;
          const emotion = node.affect?.primary_emotion || '';
          if (emotion && this.hiddenEmotions.includes(emotion)) return false;
          return true;
        });
      },
      displayEdges() {
        const active = this.activeNodeSet();
        return (this.landscape.edges || []).filter(edge =>
          active.has(edge.source_node_id) && active.has(edge.target_node_id)
        );
      },
      async selectNodeById(nodeId) {
        this.selectedNodeId = nodeId;
        await this.loadNodeDetail();
        this.renderLandscapeChart();
        this.scrollTimelineToNode(nodeId);
        this.scrollDetailIntoView();
      },
      scrollTimelineToNode(nodeId) {
        const el = document.getElementById('tl-' + nodeId);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      },
      scrollDetailIntoView() {
        const el = document.querySelector('.memory-detail-panel');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      },
      findNode(nodeId) {
        return (this.landscape.nodes || []).find(node => node.node_id === nodeId) || null;
      },
      async loadNodeDetail() {
        if (!this.selectedNodeId) {
          this.nodeDetail = null;
          return;
        }
        this.nodeDetailLoading = true;
        try {
          const params = new URLSearchParams({
            node_id: this.selectedNodeId,
            limit: '18',
            min_confidence: String(this.minEdgeConfidence),
          });
          if (this.edgeKind) params.set('edge_kind', this.edgeKind);
          const response = await fetch('/api/memory/node-detail?' + params.toString());
          if (!response.ok) throw new Error('node detail unavailable');
          this.nodeDetail = await response.json();
        } catch (e) {
          console.error(e);
          this.nodeDetail = null;
        }
        this.nodeDetailLoading = false;
      },
      embeddingPeers() {
        const node = this.findNode(this.selectedNodeId);
        if (!node || typeof node.x !== 'number' || typeof node.y !== 'number') return [];
        return this.visibleNodes()
          .filter(other =>
            other.node_id !== node.node_id &&
            other.projection_group === node.projection_group &&
            typeof other.x === 'number' &&
            typeof other.y === 'number'
          )
          .map(other => ({
            ...other,
            embedding_distance: Math.hypot(other.x - node.x, other.y - node.y),
          }))
          .sort((a, b) => a.embedding_distance - b.embedding_distance)
          .slice(0, 6);
      },
      retrievalNodeIdSet() {
        return new Set((this.retrieval?.results || []).map(item => item.node_id).filter(Boolean));
      },
      nodeColor(node) {
        const emotion = node.affect?.primary_emotion || 'none';
        const emotionPalette = {
          joy: '#f59e0b',
          sadness: '#60a5fa',
          anger: '#f87171',
          fear: '#c084fc',
          surprise: '#fb7185',
          disgust: '#34d399',
          curiosity: '#38bdf8',
          neutral: '#94a3b8',
          none: '#64748b',
        };
        const kindPalette = {
          turn: '#38bdf8',
          observation: '#f97316',
          action: '#22c55e',
          compaction: '#facc15',
          consolidation: '#e879f9',
          memory: '#a78bfa',
        };
        if (this.colorBy === 'kind') {
          return kindPalette[node.kind] || '#94a3b8';
        }
        if (this.colorBy === 'embedding_model') {
          const models = Object.keys(this.landscape?.stats?.embedding_model_distribution || {});
          const index = Math.max(0, models.indexOf(node.embedding_model_id));
          return ['#38bdf8', '#34d399', '#f59e0b', '#f87171', '#a78bfa', '#fb7185'][index % 6] || '#94a3b8';
        }
        if (this.colorBy === 'salience') {
          const salience = Number(node.salience || 0);
          const hue = Math.max(0, 210 - Math.min(salience, 10) * 14);
          return `hsl(${hue} 85% 60%)`;
        }
        if (this.colorBy === 'somatic_tag') {
          return node.somatic_tag ? _hashColor(node.somatic_tag) : '#64748b';
        }
        return emotionPalette[emotion] || emotionPalette.none;
      },
      nodeRadius(node) {
        if (this.sizeBy === 'confidence') {
          const confidence = Number(node.confidence_score ?? 0.8);
          return 5 + confidence * 8;
        }
        if (this.sizeBy === 'connections') {
          return 5 + Math.min(Number(node.connection_count || 0), 12);
        }
        if (this.sizeBy === 'utility') {
          const ok = Number(node.used_successfully || 0);
          const fail = Number(node.used_unsuccessfully || 0);
          const ratio = (ok + 1) / (ok + fail + 2);
          return 4 + ratio * 12;
        }
        return 5 + Math.min(Number(node.salience || 0), 10) * 0.9;
      },
      edgeColor(edge) {
        const palette = {
          semantic: 'rgba(56, 189, 248, 0.35)',
          emotional: 'rgba(244, 114, 182, 0.35)',
          temporal: 'rgba(250, 204, 21, 0.35)',
          conceptual: 'rgba(168, 85, 247, 0.35)',
          relational: 'rgba(34, 197, 94, 0.35)',
          causal: 'rgba(248, 113, 113, 0.35)',
          distilled_from: 'rgba(148, 163, 184, 0.28)',
        };
        return palette[edge.kind] || 'rgba(148, 163, 184, 0.28)';
      },
      renderLandscapeChart() {
        const nodes = this.displayNodes();
        const canvas = document.getElementById('memoryLandscapeChart');
        if (!canvas) return;
        if (this.landscapeChart) {
          this.landscapeChart.destroy();
          this.landscapeChart = null;
        }
        const ctx = canvas.getContext('2d');
        if (!nodes.length) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          return;
        }
        const nodeIndexById = new Map(nodes.map((node, index) => [node.node_id, index]));
        const edges = String(this.showEdges) === 'true'
          ? this.displayEdges().filter(edge =>
              nodeIndexById.has(edge.source_node_id) && nodeIndexById.has(edge.target_node_id)
            )
          : [];
        const selectedHighlights = this.retrievalNodeIdSet();
        const app = this;
        const edgePlugin = {
          id: 'memoryEdges',
          afterDatasetsDraw(chart) {
            if (!edges.length) return;
            const meta = chart.getDatasetMeta(0);
            const chartPoints = meta.data || [];
            const context = chart.ctx;
            context.save();
            edges.forEach(edge => {
              const source = chartPoints[nodeIndexById.get(edge.source_node_id)];
              const target = chartPoints[nodeIndexById.get(edge.target_node_id)];
              if (!source || !target) return;
              context.beginPath();
              context.strokeStyle = app.edgeColor(edge);
              context.lineWidth = 0.6 + Math.max(Number(edge.strength || 0), 0) * 2.2;
              context.moveTo(source.x, source.y);
              context.lineTo(target.x, target.y);
              context.stroke();
            });
            context.restore();
          }
        };
        const haloPlugin = {
          id: 'memoryIdentityHalo',
          afterDatasetsDraw(chart) {
            const meta = chart.getDatasetMeta(0);
            const chartPoints = meta.data || [];
            const context = chart.ctx;
            context.save();
            nodes.forEach((node, index) => {
              if (!node.identity_core) return;
              const point = chartPoints[index];
              if (!point) return;
              const radius = Number(chart.data.datasets[0].pointRadius[index] || 6);
              context.beginPath();
              context.strokeStyle = 'rgba(250, 204, 21, 0.8)';
              context.lineWidth = node.node_id === app.selectedNodeId ? 3 : 2;
              context.arc(point.x, point.y, radius + 4, 0, Math.PI * 2);
              context.stroke();
            });
            context.restore();
          }
        };
        const hasOverlay = this.showRetrievalOnAtlas && selectedHighlights.size > 0;
        this.landscapeChart = new Chart(ctx, {
          type: 'scatter',
          plugins: [edgePlugin, haloPlugin],
          data: {
            datasets: [{
              label: 'Memory atlas',
              data: nodes.map(node => ({ x: node.x, y: node.y })),
              backgroundColor: nodes.map(node => {
                const color = this.nodeColor(node);
                if (hasOverlay && !selectedHighlights.has(node.node_id) && node.node_id !== this.selectedNodeId) {
                  return color.replace(/[\d.]+\)$/, '0.15)').replace(/#([0-9a-f]{6})/i, (_match, hex) => {
                    const r = parseInt(hex.slice(0, 2), 16);
                    const g = parseInt(hex.slice(2, 4), 16);
                    const b = parseInt(hex.slice(4, 6), 16);
                    return `rgba(${r},${g},${b},0.15)`;
                  });
                }
                return color;
              }),
              pointRadius: nodes.map(node => {
                const base = this.nodeRadius(node);
                if (hasOverlay && selectedHighlights.has(node.node_id)) return base * 1.4;
                return base;
              }),
              pointBorderWidth: nodes.map(node => {
                if (node.node_id === this.selectedNodeId) return 3;
                if (hasOverlay && selectedHighlights.has(node.node_id)) return 2.5;
                return selectedHighlights.has(node.node_id) ? 2 : 1;
              }),
              pointBorderColor: nodes.map(node => {
                if (node.node_id === this.selectedNodeId) return '#f8fafc';
                if (hasOverlay && selectedHighlights.has(node.node_id)) return '#facc15';
                return selectedHighlights.has(node.node_id) ? '#facc15' : 'rgba(15, 23, 42, 0.85)';
              }),
              pointStyle: nodes.map(node => {
                if (node.node_type === 'memory') return 'rectRounded';
                if (node.kind === 'action') return 'triangle';
                if (node.kind === 'compaction') return 'rect';
                if (node.kind === 'consolidation') return 'rectRot';
                return 'circle';
              }),
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            onClick: (_event, elements) => {
              if (!elements.length) return;
              const node = nodes[elements[0].index];
              if (!node) return;
              this.selectNodeById(node.node_id);
            },
            plugins: {
              legend: { display: false },
              zoom: {
                zoom: {
                  wheel: { enabled: true },
                  pinch: { enabled: true },
                  drag: { enabled: true },
                  mode: 'xy',
                },
                pan: {
                  enabled: true,
                  mode: 'xy',
                },
              },
              tooltip: {
                callbacks: {
                  label: (context) => {
                    const node = nodes[context.dataIndex];
                    if (!node) return '';
                    const bits = [
                      `${node.node_type}:${node.kind || node.node_type}`,
                      `salience ${Number(node.salience || 0).toFixed(2)}`,
                      `links ${node.connection_count || 0}`,
                    ];
                    if (node.affect?.primary_emotion) bits.push(`emotion ${node.affect.primary_emotion}`);
                    if (node.embedding_model_id) bits.push(node.embedding_model_id);
                    return bits;
                  }
                }
              }
            },
            scales: {
              x: { title: { display: true, text: 'Embedding lanes' }, grid: { color: 'rgba(255,255,255,0.05)' } },
              y: { title: { display: true, text: 'Local neighborhood' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
          }
        });
      },
      memoryHealthMarkup() {
        const stats = this.landscape?.stats || {};
        const nodes = this.landscape?.nodes || [];
        if (!nodes.length && !this.memoryValue) {
          return '<p class="muted">Load the atlas to surface compaction ratio, identity-core density, and retrieval evidence.</p>';
        }
        const episodeCount = Number(stats.visible_episode_count || 0);
        const memoryCount = Number(stats.visible_memory_count || 0);
        const compactedCount = nodes.filter(node => node.compacted).length;
        const identityCoreCount = nodes.filter(node => node.identity_core).length;
        const avgSalience = nodes.length
          ? (nodes.reduce((sum, node) => sum + Number(node.salience || 0), 0) / nodes.length)
          : 0;
        const compactionRatio = episodeCount > 0 ? compactedCount / episodeCount : 0;
        const identityRatio = nodes.length > 0 ? identityCoreCount / nodes.length : 0;
        const affectEntries = Object.entries(stats.emotion_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
        const value = this.memoryValue || {};
        let html = `<div class="memory-health-grid">
          <div>
            <div class="health-header">
              <span class="badge ${value.evidence_level === 'grounded' ? 'ok' : value.evidence_level === 'partial' ? 'warn' : 'fail'}">${escapeHtml(value.evidence_level || 'atlas-only')}</span>
              <span class="muted">memory health snapshot</span>
            </div>
            <div class="stat-grid">
              <div><div class="stat-value">${episodeCount}</div><div class="stat-label">Episodes</div></div>
              <div><div class="stat-value">${memoryCount}</div><div class="stat-label">Memories</div></div>
              <div><div class="stat-value">${identityCoreCount}</div><div class="stat-label">Identity Core</div></div>
              <div><div class="stat-value">${avgSalience.toFixed(2)}</div><div class="stat-label">Avg Salience</div></div>
            </div>
            <div class="memory-health-meters mt-3">
              <div>
                <div class="helper-line"><span>Compaction ratio</span><strong>${Math.round(compactionRatio * 100)}%</strong></div>
                <div class="memory-health-bar"><span style="width:${Math.round(compactionRatio * 100)}%"></span></div>
              </div>
              <div>
                <div class="helper-line"><span>Identity-core density</span><strong>${Math.round(identityRatio * 100)}%</strong></div>
                <div class="memory-health-bar accent-gold"><span style="width:${Math.round(identityRatio * 100)}%"></span></div>
              </div>
            </div>
          </div>
          <div>
            <h5>Affect distribution</h5>
            ${affectEntries.length ? '<div class="memory-health-stack">' + affectEntries.map(([emotion, count]) => {
              const pct = nodes.length ? Math.max(8, Math.round((count / nodes.length) * 100)) : 0;
              return `<button class="legend-chip ${this.hiddenEmotions.includes(emotion) ? 'off' : ''}" onclick="window.__openCASMemoryApp.toggleEmotionVisibility('${escapeHtml(emotion)}')"><span>${escapeHtml(emotion)}</span><strong>${count}</strong><div class="memory-health-bar"><span style="width:${pct}%"></span></div></button>`;
            }).join('') + '</div>' : '<p class="muted">No affect-bearing nodes in the current atlas scope.</p>'}
            <p class="helper-text mt-3">Top affect lanes can be toggled directly from here; hidden lanes dim from the atlas until re-enabled.</p>
          </div>
        </div>`;
        return html;
      },
      landscapeSummaryMarkup() {
        const stats = this.landscape?.stats || {};
        const projectionGroups = this.landscape?.projection?.groups || [];
        const nodes = this.landscape?.nodes || [];
        if (!nodes.length) {
          return '<p class="muted">Load the atlas to see memory density, embedding families, and edge coverage.</p>';
        }
        const identityCoreCount = nodes.filter(n => n.identity_core).length;
        const avgSalience = nodes.length ? (nodes.reduce((s, n) => s + Number(n.salience || 0), 0) / nodes.length).toFixed(2) : '-';
        const kindDist = stats.kind_distribution || {};
        let html = `<div class="stat-grid">
          <div><div class="stat-value">${stats.visible_episode_count || 0}</div><div class="stat-label">Episodes</div></div>
          <div><div class="stat-value">${stats.visible_memory_count || 0}</div><div class="stat-label">Memories</div></div>
          <div><div class="stat-value">${stats.visible_edge_count || 0}</div><div class="stat-label">Edges</div></div>
          <div><div class="stat-value">${Number(stats.time_span_days || 0).toFixed(1)}d</div><div class="stat-label">Time Span</div></div>
          <div><div class="stat-value">${Number(stats.freshest_visible_age_days || 0).toFixed(1)}d</div><div class="stat-label">Freshest</div></div>
          <div><div class="stat-value">${Number(stats.average_edge_strength || 0).toFixed(2)}</div><div class="stat-label">Avg Edge Strength</div></div>
          <div><div class="stat-value">${identityCoreCount} <span class="badge-star">★</span></div><div class="stat-label">Identity Core</div></div>
          <div><div class="stat-value">${avgSalience}</div><div class="stat-label">Avg Salience</div></div>
          <div><div class="stat-value">${stats.embeddingless_node_count || 0}</div><div class="stat-label">No Embedding</div></div>
          <div><div class="stat-value">${this.displayNodes().length}</div><div class="stat-label">Visible Nodes</div></div>
        </div>`;
        html += `<div class="pill-row mt-3">`;
        projectionGroups.forEach(group => {
          const label = group.dimension ? `${group.dimension}d` : 'no embedding';
          html += `<span class="badge">${escapeHtml(label)} • ${group.count}</span>`;
        });
        html += `</div>`;
        const kindEntries = Object.entries(kindDist).sort((a, b) => b[1] - a[1]);
        if (kindEntries.length) {
          html += `<div class="pill-row mt-2">`;
          kindEntries.forEach(([kind, count]) => {
            html += `<span class="badge">${escapeHtml(kind)}: ${count}</span>`;
          });
          html += `</div>`;
        }
        html += `<div class="helper-text mt-3">Projection: <strong>${escapeHtml(stats.projection_method || '-')}</strong> • Edge floor: ${escapeHtml(String(stats.min_edge_confidence ?? '-'))} • Edge kind: ${escapeHtml(stats.edge_kind || 'all')}</div>`;
        return html;
      },
      atlasLegendMarkup() {
        const kindEntries = Object.entries(this.landscape?.stats?.kind_distribution || {}).sort((a, b) => b[1] - a[1]);
        const emotionEntries = Object.entries(this.landscape?.stats?.emotion_distribution || {}).sort((a, b) => b[1] - a[1]);
        const groups = (this.landscape?.projection?.groups || []).map(group => {
          const label = group.dimension ? `${group.dimension}d` : 'no embedding';
          return `<span class="badge">${escapeHtml(label)} • ${group.count}</span>`;
        }).join('');
        let html = `<div class="legend-chip-row">
          <span class="badge">Color ${escapeHtml(this.colorBy)}</span>
          <span class="badge">Size ${escapeHtml(this.sizeBy)}</span>
          <span class="badge">View ${escapeHtml(this.viewMode)}</span>
          ${groups || '<span class="badge">no projection groups</span>'}
          ${(this.hiddenKinds.length || this.hiddenEmotions.length) ? '<button class="btn-link" onclick="window.__openCASMemoryApp.clearLegendFilters()">Clear atlas filters</button>' : ''}
        </div>`;
        if (kindEntries.length) {
          html += '<div class="legend-chip-row mt-2">';
          kindEntries.forEach(([kind, count]) => {
            html += `<button class="legend-chip ${this.hiddenKinds.includes(kind) ? 'off' : ''}" onclick="window.__openCASMemoryApp.toggleKindVisibility('${escapeHtml(kind)}')"><span>${escapeHtml(kind)}</span><strong>${count}</strong></button>`;
          });
          html += '</div>';
        }
        if (emotionEntries.length) {
          html += '<div class="legend-chip-row mt-2">';
          emotionEntries.forEach(([emotion, count]) => {
            html += `<button class="legend-chip ${this.hiddenEmotions.includes(emotion) ? 'off' : ''}" onclick="window.__openCASMemoryApp.toggleEmotionVisibility('${escapeHtml(emotion)}')"><span>${escapeHtml(emotion)}</span><strong>${count}</strong></button>`;
          });
          html += '</div>';
        }
        return html;
      },
      resetZoom() {
        if (this.landscapeChart && typeof this.landscapeChart.resetZoom === 'function') {
          this.landscapeChart.resetZoom();
        }
      },
      selectedNodeMarkup() {
        const node = this.findNode(this.selectedNodeId);
        if (!node) {
          return '<p class="muted">Select a node from the atlas to inspect its content, affect, and strongest connections.</p>';
        }
        const detail = this.nodeDetail;
        const detailEdges = detail?.edges || [];
        const detailStats = detail?.stats || {};
        const neighbors = detailEdges
          .map(edge => {
            const otherId = edge.other_node_id || (edge.source_node_id === node.node_id ? edge.target_node_id : edge.source_node_id);
            const otherNode = (detail?.neighbors || []).find(item => item.node_id === otherId) || this.findNode(otherId);
            return { edge, node: otherNode };
          })
          .filter(item => item.node)
          .slice(0, 8);
        const connectionMix = {};
        ((this.landscape.edges || []).filter(edge => edge.source_node_id === node.node_id || edge.target_node_id === node.node_id)).forEach(edge => {
          connectionMix[edge.kind] = (connectionMix[edge.kind] || 0) + 1;
        });
        const connectionMixText = Object.entries(connectionMix)
          .sort((a, b) => b[1] - a[1])
          .map(([kind, count]) => `${kind} ${count}`)
          .join(' • ');
        const peers = this.embeddingPeers();
        let html = `<div class="stack">
          <div class="pill-row">
            <span class="badge">${escapeHtml(node.node_type)}</span>
            <span class="badge">${escapeHtml(node.kind || '-')}</span>
            ${node.affect?.primary_emotion ? `<span class="badge">${escapeHtml(node.affect.primary_emotion)}</span>` : ''}
            ${node.identity_core ? '<span class="badge badge-gold">★ identity core</span>' : ''}
            ${node.compacted ? '<span class="badge badge-dim">compacted</span>' : ''}
          </div>
          ${node.session_id ? `<div class="mt-2"><button class="btn-link session-chip" onclick="window.__openCASMemoryApp.filterBySession('${escapeHtml(node.session_id)}')">${escapeHtml(node.session_id)}</button></div>` : ''}
          <p class="muted">Created: ${formatDateTime(node.created_at)} • Age: ${escapeHtml(String(node.age_days ?? '-'))}d</p>
          <p class="muted">Embedding: ${escapeHtml(node.embedding_model_id || 'none')} • Group: ${escapeHtml(node.projection_group || '-')}</p>
          <p class="muted">Salience: ${escapeHtml(String(node.salience ?? '-'))} • Confidence: ${escapeHtml(String(node.confidence_score ?? '-'))} • Connections: ${escapeHtml(String(node.connection_count ?? 0))}</p>
          ${node.somatic_tag ? `<p class="muted">Somatic tag: <span class="badge">${escapeHtml(node.somatic_tag)}</span></p>` : ''}
          ${(node.used_successfully > 0 || node.used_unsuccessfully > 0) ? (() => {
            const total = node.used_successfully + node.used_unsuccessfully;
            const pct = Math.round((node.used_successfully / total) * 100);
            return `<div class="utility-bar-wrap"><span class="muted">Utility:</span><div class="utility-bar"><div class="utility-bar-fill" style="width:${pct}%"></div></div><span class="muted">${node.used_successfully}✓ / ${node.used_unsuccessfully}✗</span></div>`;
          })() : ''}
          ${node.last_accessed ? `<p class="muted">Last accessed: ${formatDateTime(node.last_accessed)}</p>` : ''}
          <div class="json">${escapeHtml(node.content || '')}</div>`;
        html += `<p class="muted mt-3">Connection mix: ${escapeHtml(connectionMixText || 'none')}</p>`;
        if (this.nodeDetailLoading) {
          html += `<p class="muted">Loading neighborhood detail…</p>`;
        } else if (detail) {
          html += `<div class="stat-grid mt-3">
            <div><div class="stat-value">${detailStats.neighbor_count || 0}</div><div class="stat-label">Neighbors</div></div>
            <div><div class="stat-value">${detailStats.edge_count || 0}</div><div class="stat-label">Edges</div></div>
          </div>`;
        }
        if (node.affect) {
          html += `<div class="memory-score-grid mt-3">
            <div><span class="muted">valence</span><strong>${escapeHtml(String(node.affect.valence ?? '-'))}</strong></div>
            <div><span class="muted">arousal</span><strong>${escapeHtml(String(node.affect.arousal ?? '-'))}</strong></div>
            <div><span class="muted">intensity</span><strong>${escapeHtml(String(node.affect.intensity ?? '-'))}</strong></div>
            <div><span class="muted">certainty</span><strong>${escapeHtml(String(node.affect.certainty ?? '-'))}</strong></div>
          </div>`;
        }
        if (node.node_type === 'memory' && (node.source_episode_ids || []).length) {
          html += `<h5 class="mt-4">Source Episodes</h5><div class="pill-row">`;
          (node.source_episode_ids || []).forEach(sourceId => {
            html += `<button class="btn-link" onclick="window.__openCASMemoryApp.selectNodeById('episode:${escapeHtml(sourceId)}')">${escapeHtml(sourceId)}</button>`;
          });
          html += `</div>`;
        }
        if (peers.length) {
          html += '<h5 class="mt-4">Nearest Embedding Peers</h5><table class="data-table"><thead><tr><th>Peer</th><th>Distance</th><th>Kind</th></tr></thead><tbody>';
          peers.forEach(peer => {
            html += `<tr>
              <td><button class="btn-link" onclick="window.__openCASMemoryApp.selectNodeById('${escapeHtml(peer.node_id)}')">${escapeHtml(peer.label || peer.node_id)}</button></td>
              <td>${escapeHtml(Number(peer.embedding_distance || 0).toFixed(3))}</td>
              <td>${escapeHtml(peer.kind || peer.node_type || '-')}</td>
            </tr>`;
          });
          html += '</tbody></table>';
        }
        if (neighbors.length) {
          html += '<h5 class="mt-4">Strongest Links</h5><table class="data-table"><thead><tr><th>Neighbor</th><th>Kind</th><th>Strength</th><th>Dominant Signal</th><th>Time Gap</th></tr></thead><tbody>';
          neighbors.forEach(({ edge, node: other }) => {
            html += `<tr>
              <td><button class="btn-link" onclick="window.__openCASMemoryApp.selectNodeById('${escapeHtml(other?.node_id || '')}')">${escapeHtml(other?.label || other?.node_id || '-')}</button></td>
              <td>${escapeHtml(edge.kind || '-')}</td>
              <td>${escapeHtml(String(edge.strength ?? '-'))}</td>
              <td>${escapeHtml(edge.strongest_signal || '-')} ${edge.strongest_signal_weight != null ? escapeHtml(Number(edge.strongest_signal_weight).toFixed(2)) : ''}</td>
              <td>${escapeHtml(String(edge.time_distance_days ?? '-'))}</td>
            </tr>`;
          });
          html += '</tbody></table>';
        }
        return html + '</div>';
      },
      memoryValueMarkup() {
        if (this.memoryValueLoading) {
          return '<p class="muted">Loading memory value evidence…</p>';
        }
        if (!this.memoryValue) {
          return '<p class="muted">Memory value evidence is not available in the current runtime.</p>';
        }
        return renderMemoryValue(this.memoryValue);
      },
      episodeListMarkup() {
        const nodes = (this.landscape.nodes || [])
          .filter(node => node.node_type === 'episode')
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        if (!nodes.length) {
          return '<p class="muted">No episode nodes in the current atlas scope.</p>';
        }
        let html = '<div class="episode-list">';
        nodes.forEach(node => {
          html += `<div id="tl-${escapeHtml(node.node_id)}" class="episode ${node.node_id === this.selectedNodeId ? 'episode-active' : ''}">
            <div class="episode-header">
              <span class="badge">${escapeHtml(node.kind || 'episode')}</span>
              ${node.identity_core ? '<span class="badge badge-gold">★</span>' : ''}
              ${node.compacted ? '<span class="badge badge-dim">compacted</span>' : ''}
              <span class="muted">${formatDateTime(node.created_at)}</span>
              <span class="muted">links ${escapeHtml(String(node.connection_count || 0))}</span>
            </div>
            <div class="episode-body">${escapeHtml(node.content || '')}</div>
            <div class="episode-footer muted">
              ${node.affect?.primary_emotion ? `<span class="badge">${escapeHtml(node.affect.primary_emotion)}</span> • ` : ''}
              salience ${escapeHtml(String(node.salience ?? '-'))} • ${escapeHtml(node.embedding_model_id || 'no embedding')}
              ${node.session_id ? ` • <button class="btn-link session-chip" onclick="window.__openCASMemoryApp.filterBySession('${escapeHtml(node.session_id)}')">${escapeHtml(node.session_id.slice(0,8))}…</button>` : ''}
            </div>
            <div class="inline-actions mt-2">
              <button class="btn-link" onclick="window.__openCASMemoryApp.selectNodeById('${escapeHtml(node.node_id)}')">Focus in atlas</button>
            </div>
          </div>`;
        });
        return html + '</div>';
      },
      somaticMarkup() {
        if (this.somaticLoading || this.musubiLoading) {
          return '<p class="muted">Loading somatic and relational state…</p>';
        }
        const s = this.somatic;
        const m = this.musubiState;
        if (!s && !m) {
          return '<p class="muted">Somatic and relational state are not available in the current runtime.</p>';
        }
        let html = '<div class="somatic-relational-grid">';

        if (s) {
          const salienceMod = 1.0 + (s.arousal * 0.3) + (s.tension * 0.3) - (s.fatigue * 0.2);
          const vitals = [
            { label: 'arousal',   value: s.arousal,   color: '#f59e0b', range: [0,1] },
            { label: 'energy',    value: s.energy,    color: '#22c55e', range: [0,1] },
            { label: 'focus',     value: s.focus,     color: '#38bdf8', range: [0,1] },
            { label: 'certainty', value: s.certainty, color: '#a78bfa', range: [0,1] },
            { label: 'fatigue',   value: s.fatigue,   color: '#94a3b8', range: [0,1] },
            { label: 'tension',   value: s.tension,   color: '#f87171', range: [0,1] },
            { label: 'valence',   value: s.valence,   color: s.valence >= 0 ? '#34d399' : '#f87171', range: [-1,1] },
          ];
          html += `<div class="somatic-panel">
            <h5>Somatic State</h5>
            <div class="pill-row mb-2">
              ${s.primary_emotion ? `<span class="badge">${escapeHtml(s.primary_emotion)}</span>` : ''}
              ${s.somatic_tag ? `<span class="badge">${escapeHtml(s.somatic_tag)}</span>` : ''}
              <span class="muted" style="font-size:11px">updated ${escapeHtml(s.updated_at ? s.updated_at.replace('T',' ').slice(0,19) : '-')}</span>
            </div>
            <div class="somatic-bars">
              ${vitals.map(v => {
                const [lo, hi] = v.range;
                const pct = Math.round(((v.value - lo) / (hi - lo)) * 100);
                const sign = v.range[0] < 0 && v.value >= 0 ? '+' : '';
                return `<div class="somatic-bar-row">
                  <span class="somatic-bar-label">${escapeHtml(v.label)}</span>
                  <div class="somatic-bar-track"><div class="somatic-bar-fill" style="width:${pct}%;background:${v.color}"></div></div>
                  <span class="somatic-bar-value">${sign}${Number(v.value).toFixed(2)}</span>
                </div>`;
              }).join('')}
            </div>
            <div class="helper-text mt-3">
              <strong>Salience modifier now:</strong> ${salienceMod.toFixed(3)}
              <span class="muted"> = 1 + arousal×0.3 + tension×0.3 − fatigue×0.2</span>
            </div>
          </div>`;
        }

        if (m) {
          const dims = [
            { label: 'trust',      value: m.dimensions.trust      || 0, color: '#38bdf8' },
            { label: 'resonance',  value: m.dimensions.resonance  || 0, color: '#fb7185' },
            { label: 'presence',   value: m.dimensions.presence   || 0, color: '#34d399' },
            { label: 'attunement', value: m.dimensions.attunement || 0, color: '#a78bfa' },
          ];
          const musubiPct = Math.round(((m.musubi + 1) / 2) * 100);
          const musubiColor = m.musubi >= 0.4 ? '#34d399' : m.musubi >= 0 ? '#f59e0b' : '#f87171';
          const relMult = (1 + m.musubi * 0.2).toFixed(3);
          html += `<div class="musubi-panel">
            <h5>Relational State (Musubi)</h5>
            <div class="pill-row mb-2">
              ${m.source_tag ? `<span class="badge">${escapeHtml(m.source_tag)}</span>` : ''}
              <span class="muted" style="font-size:11px">updated ${escapeHtml(m.updated_at ? m.updated_at.replace('T',' ').slice(0,19) : '-')}</span>
            </div>
            <div class="somatic-bar-row mb-3" style="gap:10px">
              <span class="somatic-bar-label"><strong>musubi</strong></span>
              <div class="somatic-bar-track"><div class="somatic-bar-fill" style="width:${musubiPct}%;background:${musubiColor}"></div></div>
              <span class="somatic-bar-value" style="color:${musubiColor}">${m.musubi >= 0 ? '+' : ''}${Number(m.musubi).toFixed(3)}</span>
            </div>
            <div class="somatic-bars">
              ${dims.map(d => {
                const pct = Math.round(((d.value + 1) / 2) * 100);
                return `<div class="somatic-bar-row">
                  <span class="somatic-bar-label">${escapeHtml(d.label)}</span>
                  <div class="somatic-bar-track"><div class="somatic-bar-fill" style="width:${pct}%;background:${d.color}"></div></div>
                  <span class="somatic-bar-value">${d.value >= 0 ? '+' : ''}${Number(d.value).toFixed(2)}</span>
                </div>`;
              }).join('')}
            </div>
            <div class="helper-text mt-3">
              <strong>Retrieval multiplier now:</strong> ${relMult}
              <span class="muted"> = 1 + musubi×0.2 (applied per candidate)</span>
            </div>
            ${m.continuity_breadcrumb ? `<p class="muted mt-2" style="font-style:italic">"${escapeHtml(m.continuity_breadcrumb)}"</p>` : ''}
          </div>`;
        }

        const taggedNodes = (this.landscape.nodes || []).filter(n => n.somatic_tag);
        if (taggedNodes.length) {
          const tagCounts = {};
          taggedNodes.forEach(n => { tagCounts[n.somatic_tag] = (tagCounts[n.somatic_tag] || 0) + 1; });
          const tagEntries = Object.entries(tagCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
          html += `<div class="somatic-tags-panel">
            <h5>Somatic Tags in Current Atlas</h5>
            <p class="muted">Emotional state recorded at memory formation time — a trace of somatic history through the graph.</p>
            <div class="pill-row mt-2">
              ${tagEntries.map(([tag, count]) => `<span class="badge" style="border-left:3px solid ${_hashColor(tag)}">${escapeHtml(tag)} <strong>${count}</strong></span>`).join('')}
            </div>
          </div>`;
        }

        html += '</div>';
        return html;
      },
      async inspectRetrieval() {
        if (!this.retrievalQuery.trim()) return;
        this.retrievalLoading = true;
        try {
          const params = new URLSearchParams({
            query: this.retrievalQuery.trim(),
            limit: String(this.retrievalLimit),
            min_confidence: String(this.retrievalMinConfidence),
            lambda_param: String(this.retrievalLambda),
            expand_graph: String(String(this.retrievalExpandGraph) === 'true'),
            semantic_weight: String(this.weights.semantic_score),
            keyword_weight: String(this.weights.keyword_score),
            recency_weight: String(this.weights.recency_score),
            salience_weight: String(this.weights.salience_score),
            graph_weight: String(this.weights.graph_score),
            emotional_weight: String(this.weights.emotional_resonance),
            temporal_weight: String(this.weights.temporal_echo),
            reliability_weight: String(this.weights.reliability),
          });
          if (this.sessionId.trim()) params.set('session_id', this.sessionId.trim());
          const response = await fetch('/api/memory/retrieval-inspect?' + params.toString());
          this.retrieval = await response.json();
          const firstNodeId = this.retrieval?.results?.[0]?.node_id;
          if (firstNodeId) {
            await this.selectNodeById(firstNodeId);
          }
          this.renderLandscapeChart();
        } catch (e) {
          console.error(e);
        }
        this.retrievalLoading = false;
      },
      retrievalMarkup() {
        if (this.retrievalLoading) {
          return '<p class="muted">Inspecting retrieval…</p>';
        }
        if (!this.retrieval) {
          return '<p class="muted">Run a retrieval query to inspect score breakdowns and selected memories.</p>';
        }
        const results = this.retrieval.results || [];
        const candidates = (this.retrieval.candidates || []).slice(0, 18);
        let html = `<div class="stat-grid">
          <div><div class="stat-value">${results.length}</div><div class="stat-label">Returned</div></div>
          <div><div class="stat-value">${this.retrieval.meta?.semantic_seed_count || 0}</div><div class="stat-label">Semantic Seeds</div></div>
          <div><div class="stat-value">${this.retrieval.meta?.keyword_seed_count || 0}</div><div class="stat-label">Keyword Seeds</div></div>
          <div><div class="stat-value">${candidates.length}</div><div class="stat-label">Top Candidates</div></div>
        </div>`;
        if (results.length) {
          html += '<h5 class="mt-4">Selected Results</h5><table class="data-table"><thead><tr><th>Memory</th><th>Type</th><th>Score</th><th>Action</th></tr></thead><tbody>';
          results.forEach(item => {
            html += `<tr>
              <td>${escapeHtml(item.content_preview || item.content || '-')}</td>
              <td>${escapeHtml(item.source_type || '-')}</td>
              <td>${escapeHtml(String(item.score ?? '-'))}</td>
              <td><button class="btn-link" onclick="window.__openCASMemoryApp.selectNodeById('${escapeHtml(item.node_id || '')}')">Focus</button></td>
            </tr>`;
          });
          html += '</tbody></table>';
        }
        html += '<h5 class="mt-4">Candidate Breakdown</h5><table class="data-table"><thead><tr><th>Candidate</th><th>Signals</th><th>Score Path</th><th>Selected</th></tr></thead><tbody>';
        candidates.forEach(item => {
          const signals = [
            `sem ${Number(item.semantic_score || 0).toFixed(2)}`,
            `key ${Number(item.keyword_score || 0).toFixed(2)}`,
            `graph ${Number(item.graph_score || 0).toFixed(2)}`,
            `emo ${Number(item.emotional_resonance || 0).toFixed(2)}`,
          ].join(' • ');
          const scorePath = [
            `base ${Number(item.base_score || 0).toFixed(2)}`,
            `somatic ${Number(item.somatic_bonus || 0).toFixed(2)}`,
            `× rel ${Number(item.reliability_multiplier || 0).toFixed(2)}`,
            `× musubi ${Number(item.relational_multiplier || 0).toFixed(2)}`,
            `× conf ${Number(item.confidence_multiplier || 0).toFixed(2)}`,
            `= ${Number(item.final_score || 0).toFixed(2)}`,
          ].join(' • ');
          html += `<tr>
            <td><button class="btn-link" onclick="window.__openCASMemoryApp.selectNodeById('${escapeHtml(item.source_type + ':' + item.source_id)}')">${escapeHtml(_truncateText(item.content || '-', 88))}</button></td>
            <td>${escapeHtml(signals)}</td>
            <td>${escapeHtml(scorePath)}</td>
            <td>${item.selected ? '<span class="badge ok">yes</span>' : '<span class="badge">no</span>'}</td>
          </tr>`;
        });
        html += '</tbody></table>';
        return html;
      }
    };
  }

  function renderProjection(d) {
    const points = d.points || [];
    const method = d.method || 'none';
    let html = '<h4>Embedding Projection</h4>';
    if (!points.length) return html + '<p class="muted">No projection data available.</p>';
    html += '<div style="position:relative;max-height:50vh;"><canvas id="projectionChart"></canvas></div>';
    html += '<p class="muted mt-2">Method: ' + escapeHtml(method) + ' • Points: ' + points.length + '</p>';
    setTimeout(() => {
      const canvas = document.getElementById('projectionChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (global.__projectionChartInstance) {
        global.__projectionChartInstance.destroy();
      }
      const colors = {
        user: '#60a5fa',
        assistant: '#34d399',
        system: '#fbbf24',
        memory: '#a78bfa',
        tool: '#f87171',
        episode: '#94a3b8'
      };
      global.__projectionChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets: [{
            label: 'Episodes',
            data: points.map(p => ({ x: p.x, y: p.y })),
            backgroundColor: points.map(p => colors[p.kind] || colors.episode),
            pointRadius: points.map(p => 3 + (p.salience || 0.5) * 5),
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const p = points[context.dataIndex];
                  return p.kind + ' | salience ' + (p.salience ?? '-');
                }
              }
            }
          },
          scales: {
            x: { title: { display: true, text: 'X' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { title: { display: true, text: 'Y' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          }
        }
      });
    }, 0);
    return html;
  }

  global.renderMemoryStats = renderMemoryStats;
  global.memoryApp = memoryApp;
  global.renderProjection = renderProjection;
})(window);
