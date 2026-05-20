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
      includeContextProposals: 'true',
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
      memoryActivity: { active_nodes: [], active_node_ids: [], events: [], stats: {} },
      memoryActivityLoading: false,
      memoryActivityTimer: null,
      cognitiveFlow: { stats: {}, nodes: [], edges: [] },
      cognitiveFlowLoading: false,
      cognitiveFlowTimer: null,
      cognitiveFlowRenderFrame: null,
      cognitiveFlowResizeHandler: null,
      cognitiveFlowMode: 'projection',
      cognitiveFlowShowSpanFlow: true,
      cognitiveFlowShowSequential: true,
      cognitiveFlowWindowSeconds: 180,
      cognitiveFlowLimit: 120,
      cognitiveFlowKindFilter: '',
      cognitiveFlowIncludeNoise: false,
      cognitiveFlowSelectedNodeId: '',
      cognitiveFlowNodePositions: [],
      atlasExpanded: false,
      initMemory() {
        global.__openCASMemoryApp = this;
        this.loadGlobalStats();
        this.loadMemoryValue();
        this.loadLandscape();
        this.loadSomaticState();
        this.loadMusubiState();
        this.startMemoryActivityPolling();
        this.startCognitiveFlowPolling();
        this.cognitiveFlowResizeHandler = () => {
          if (this.cognitiveFlowMode === 'flow') {
            this.scheduleCognitiveFlowRender();
          }
        };
        window.addEventListener('resize', this.cognitiveFlowResizeHandler);
      },
      destroy() {
        if (this.memoryActivityTimer) {
          clearInterval(this.memoryActivityTimer);
          this.memoryActivityTimer = null;
        }
        if (this.cognitiveFlowTimer) {
          clearInterval(this.cognitiveFlowTimer);
          this.cognitiveFlowTimer = null;
        }
        if (this.cognitiveFlowRenderFrame) {
          cancelAnimationFrame(this.cognitiveFlowRenderFrame);
          this.cognitiveFlowRenderFrame = null;
        }
        if (this.cognitiveFlowResizeHandler) {
          window.removeEventListener('resize', this.cognitiveFlowResizeHandler);
          this.cognitiveFlowResizeHandler = null;
        }
      },
      startMemoryActivityPolling() {
        if (this.memoryActivityTimer) clearInterval(this.memoryActivityTimer);
        this.loadMemoryActivity();
        this.memoryActivityTimer = setInterval(() => this.loadMemoryActivity(), 2500);
      },
      startCognitiveFlowPolling() {
        if (this.cognitiveFlowTimer) clearInterval(this.cognitiveFlowTimer);
        this.loadCognitiveFlow();
        this.cognitiveFlowTimer = setInterval(() => this.loadCognitiveFlow(), 3000);
      },
      setCognitiveFlowMode(mode) {
        this.cognitiveFlowMode = mode;
        if (mode === 'flow') {
          const nodes = this.cognitiveFlow?.nodes || [];
          if (!this.cognitiveFlowSelectedNodeId && nodes.length) {
            this.cognitiveFlowSelectedNodeId = nodes[nodes.length - 1]?.event_id || '';
          } else if (this.cognitiveFlowSelectedNodeId) {
            const found = nodes.some(node => node.event_id === this.cognitiveFlowSelectedNodeId);
            if (!found) {
              this.cognitiveFlowSelectedNodeId = nodes.length ? nodes[nodes.length - 1]?.event_id : '';
            }
          }
          this.scheduleCognitiveFlowRender();
        } else {
          this.renderLandscapeChart();
        }
      },
      toggleAtlasExpanded() {
        this.atlasExpanded = !this.atlasExpanded;
        if (this.cognitiveFlowMode === 'flow') {
          this.scheduleCognitiveFlowRender();
        } else {
          setTimeout(() => this.renderLandscapeChart(), 80);
        }
      },
      scheduleCognitiveFlowRender() {
        if (this.cognitiveFlowRenderFrame) {
          cancelAnimationFrame(this.cognitiveFlowRenderFrame);
          this.cognitiveFlowRenderFrame = null;
        }
        this.cognitiveFlowRenderFrame = requestAnimationFrame(() => {
          this.cognitiveFlowRenderFrame = requestAnimationFrame(() => {
            this.cognitiveFlowRenderFrame = null;
            this.renderCognitiveFlow();
          });
        });
      },
      openCognitiveFlowPopout() {
        const params = new URLSearchParams({
          window_seconds: String(this.cognitiveFlowWindowSeconds),
          limit: String(this.cognitiveFlowLimit),
          include_noise: String(this.cognitiveFlowIncludeNoise),
          show_span_flow: String(this.cognitiveFlowShowSpanFlow),
          show_sequential_links: String(this.cognitiveFlowShowSequential),
          mode: 'flow',
        });
        if (this.sessionId) params.set('session_id', this.sessionId);
        if (this.cognitiveFlowKindFilter.trim()) {
          params.set('kind_filter', this.cognitiveFlowKindFilter.trim());
        }
        if (this.cognitiveFlowSelectedNodeId) {
          params.set('selected_event_id', this.cognitiveFlowSelectedNodeId);
        }
        const url = `/dashboard/static/cognitive-flow.html?${params.toString()}`;
        const popout = window.open(
          url,
          'opencas-cognitive-flow',
          'noopener,noreferrer,width=1500,height=900',
        );
        if (popout) popout.focus();
      },
      cognitiveFlowNodeById(nodeId) {
        if (!nodeId) return null;
        return (this.cognitiveFlow?.nodes || []).find(node => (node.event_id || node.node_id) === nodeId) || null;
      },
      selectedCognitiveFlowNode() {
        return this.cognitiveFlowNodeById(this.cognitiveFlowSelectedNodeId);
      },
      selectCognitiveFlowNode(nodeId) {
        this.cognitiveFlowSelectedNodeId = nodeId || '';
        if (this.cognitiveFlowMode === 'flow') {
          this.renderCognitiveFlow();
        }
      },
      openAtlasNodeFromFlow(nodeRef) {
        if (!nodeRef) return;
        this.selectedNodeId = nodeRef;
        this.nodeDetail = null;
        this.selectNodeById(nodeRef);
      },
      cognitiveFlowAtlasNodeCandidates(node) {
        if (!node) return [];
        const seen = new Set();
        const candidates = [];
        const add = (nodeRef, source) => {
          if (!nodeRef) return;
          const normalized = String(nodeRef);
          if (!normalized || seen.has(normalized)) return;
          seen.add(normalized);
          candidates.push({
            node_id: normalized,
            source: source || 'linked',
            known: !!this.findNode(normalized),
          });
        };
        add(node.node_ref, 'node_ref');
        if (node.source_type && node.source_id) {
          add(`${node.source_type}:${node.source_id}`, node.source_type);
        }
        if (node.source_id && !node.source_type) {
          add(node.source_id, 'source_id');
        }
        add(node.event_payload?.memory_event_id, 'event_payload.memory_event_id');
        add(node.source_id, 'source_id');
        const payload = node.event_payload?.payload || {};
        if (typeof payload === 'object' && payload !== null) {
          add(payload.node_id, 'event_payload.payload.node_id');
          if (payload.memory_id) add(payload.memory_id, 'event_payload.payload.memory_id');
          if (payload.referenced_event_id) add(payload.referenced_event_id, 'event_payload.payload.referenced_event_id');
          if (payload.episode_id) add(`episode:${payload.episode_id}`, 'event_payload.payload.episode_id');
          if (payload.memory_event_id) add(`memory:${payload.memory_event_id}`, 'event_payload.payload.memory_event_id');
        }
        return candidates;
      },
      async loadMemoryActivity() {
        this.memoryActivityLoading = true;
        try {
          const response = await fetch('/api/memory/activity?window_seconds=180&limit=80');
          if (response.ok) {
            this.memoryActivity = await response.json();
            this.refreshLandscapeChartOverlay();
          }
        } catch (e) { console.error(e); }
        this.memoryActivityLoading = false;
      },
      async loadCognitiveFlow() {
        this.cognitiveFlowLoading = true;
        try {
          const params = new URLSearchParams({
            window_seconds: String(this.cognitiveFlowWindowSeconds),
            limit: String(this.cognitiveFlowLimit),
            include_noise: String(this.cognitiveFlowIncludeNoise),
            show_span_flow: String(this.cognitiveFlowShowSpanFlow),
            show_sequential_links: String(this.cognitiveFlowShowSequential),
          });
          if (this.sessionId.trim()) params.set('session_id', this.sessionId.trim());
          if (this.cognitiveFlowKindFilter.trim()) params.set('kind_filter', this.cognitiveFlowKindFilter.trim());
          const response = await fetch('/api/memory/cognitive-flow?' + params.toString());
          if (response.ok) {
            this.cognitiveFlow = await response.json();
            const nodes = this.cognitiveFlow?.nodes || [];
            if (!nodes.length) {
              this.cognitiveFlowSelectedNodeId = '';
            } else if (this.cognitiveFlowSelectedNodeId) {
              const keep = nodes.some(node => node.event_id === this.cognitiveFlowSelectedNodeId);
              if (!keep) {
                this.cognitiveFlowSelectedNodeId = nodes[nodes.length - 1]?.event_id || '';
              }
            } else {
              this.cognitiveFlowSelectedNodeId = nodes[nodes.length - 1]?.event_id || '';
            }
            if (this.cognitiveFlowMode === 'flow') this.scheduleCognitiveFlowRender();
          }
        } catch (e) {
          console.error(e);
          this.cognitiveFlow = { stats: { available: false }, nodes: [], edges: [] };
        }
        this.cognitiveFlowLoading = false;
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
            include_context_proposals: String(String(this.includeContextProposals) === 'true'),
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
        this.includeContextProposals = 'true';
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
        const allKinds = ['semantic', 'emotional', 'temporal', 'conceptual', 'relational', 'causal', 'distilled_from', 'proposal_evidence', 'context_proposal'];
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
      activityByNode() {
        return new Map((this.memoryActivity?.active_nodes || []).map(item => [item.node_id, item]));
      },
      activityForNode(nodeId) {
        return this.activityByNode().get(nodeId) || null;
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
          memory_association: '#f59e0b',
          project_next_step: '#22c55e',
          bad_idea_to_avoid: '#f87171',
          context_note: '#38bdf8',
        };
        if (this.colorBy === 'kind') {
          if (node.node_type === 'context_proposal') {
            return '#f59e0b';
          }
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
      nodeFillColor(node, selectedHighlights, activityByNode) {
        const color = this.nodeColor(node);
        const hasOverlay = this.showRetrievalOnAtlas && selectedHighlights.size > 0;
        if (hasOverlay && !selectedHighlights.has(node.node_id) && node.node_id !== this.selectedNodeId) {
          return color.replace(/[\d.]+\)$/, '0.15)').replace(/#([0-9a-f]{6})/i, (_match, hex) => {
            const r = parseInt(hex.slice(0, 2), 16);
            const g = parseInt(hex.slice(2, 4), 16);
            const b = parseInt(hex.slice(4, 6), 16);
            return `rgba(${r},${g},${b},0.15)`;
          });
        }
        return color;
      },
      chartNodeStyle(nodes) {
        const selectedHighlights = this.retrievalNodeIdSet();
        const activityByNode = this.activityByNode();
        const hasOverlay = this.showRetrievalOnAtlas && selectedHighlights.size > 0;
        return {
          backgroundColor: nodes.map(node => this.nodeFillColor(node, selectedHighlights, activityByNode)),
          pointRadius: nodes.map(node => {
            const base = this.nodeRadius(node);
            const activity = activityByNode.get(node.node_id);
            if (activity) return base * (1.0 + Math.min(0.65, Number(activity.intensity || 0) * 0.65));
            if (hasOverlay && selectedHighlights.has(node.node_id)) return base * 1.4;
            return base;
          }),
          pointBorderWidth: nodes.map(node => {
            if (node.node_id === this.selectedNodeId) return 3;
            if (activityByNode.has(node.node_id)) return 3;
            if (hasOverlay && selectedHighlights.has(node.node_id)) return 2.5;
            return selectedHighlights.has(node.node_id) ? 2 : 1;
          }),
          pointBorderColor: nodes.map(node => {
            if (node.node_id === this.selectedNodeId) return '#f8fafc';
            if (activityByNode.has(node.node_id)) return '#22d3ee';
            if (hasOverlay && selectedHighlights.has(node.node_id)) return '#facc15';
            return selectedHighlights.has(node.node_id) ? '#facc15' : 'rgba(15, 23, 42, 0.85)';
          }),
          pointStyle: nodes.map(node => {
            if (node.node_type === 'context_proposal') return 'star';
            if (node.node_type === 'memory') return 'rectRounded';
            if (node.kind === 'action') return 'triangle';
            if (node.kind === 'compaction') return 'rect';
            if (node.kind === 'consolidation') return 'rectRot';
            return 'circle';
          }),
        };
      },
      refreshLandscapeChartOverlay() {
        if (!this.landscapeChart) return;
        const chart = global.Alpine?.raw ? global.Alpine.raw(this.landscapeChart) : this.landscapeChart;
        const nodes = this.displayNodes();
        const dataset = chart.data.datasets[0];
        if (!dataset || dataset.data.length !== nodes.length) {
          this.renderLandscapeChart();
          return;
        }
        Object.assign(dataset, this.chartNodeStyle(nodes));
        chart.update('none');
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
          proposal_evidence: 'rgba(245, 158, 11, 0.42)',
        };
        return palette[edge.kind] || 'rgba(148, 163, 184, 0.28)';
      },
      flowColor(node) {
        const kind = node.kind || 'telemetry';
        const palette = {
          span_start: '#38bdf8',
          span_end: '#f59e0b',
          llm_call: '#f97316',
          tool_call: '#22c55e',
          memory_noted: '#e879f9',
          memory_activated: '#f59e0b',
          memory_compact: '#facc15',
          memory_write: '#a78bfa',
          action_backlink: '#ef4444',
          consolidation_run: '#fb7185',
          turn: '#34d399',
          warning: '#fcd34d',
          error: '#fb7185',
          flow_order: '#22d3ee',
          span_hierarchy: '#94a3b8',
          continuity_backlink: '#f59e0b',
          action_backlink: '#ef4444',
        };
        return palette[kind] || palette[kind.toLowerCase()] || '#94a3b8';
      },
      flowRadius(node) {
        if (node.noted_as_such) return 7;
        if (node.is_span_gate) return 8;
        return 5;
      },
      flowEdgeColor(kind) {
        const palette = {
          flow_order: 'rgba(56, 189, 248, 0.5)',
          span_hierarchy: 'rgba(168, 85, 247, 0.45)',
          lane_order: 'rgba(34, 211, 238, 0.28)',
          cognitive_sequence: 'rgba(248, 250, 252, 0.42)',
          continuity_backlink: 'rgba(34, 211, 238, 0.6)',
          action_backlink: 'rgba(248, 113, 113, 0.55)',
          default: 'rgba(148, 163, 184, 0.35)',
        };
        return palette[kind] || palette.default;
      },
      cognitiveFlowSummaryMarkup() {
        const stats = this.cognitiveFlow?.stats || {};
        if (!stats.available) {
          return '<p class="muted">Cognitive flow stream is not available in this runtime.</p>';
        }
        if (!this.cognitiveFlow?.nodes?.length) {
          return '<p class="muted">No cognitive-flow events in the selected window.</p>';
        }
        const lanes = Object.keys(stats.lane_distribution || {}).length;
        return `<div class="stat-grid">
          <div><div class="stat-value">${this.cognitiveFlow.event_count || 0}</div><div class="stat-label">Events</div></div>
          <div><div class="stat-value">${stats.edge_count || (this.cognitiveFlow.edges || []).length || 0}</div><div class="stat-label">Flow Edges</div></div>
          <div><div class="stat-value">${lanes}</div><div class="stat-label">Lanes</div></div>
          <div><div class="stat-value">${stats.readable_thought_count ?? stats.high_signal_event_count ?? 0}</div><div class="stat-label">Readable Thoughts</div></div>
          <div><div class="stat-value">${stats.excluded_noise_event_count ?? 0}</div><div class="stat-label">Hidden Noise</div></div>
          <div><div class="stat-value">${(stats.temporal_missing_node_count || 0) + (stats.temporal_missing_edge_count || 0)}</div><div class="stat-label">Missing Time Metadata</div></div>
        </div>`;
      },
      cognitiveFlowSelectionMarkup() {
        const node = this.selectedCognitiveFlowNode();
        if (!node) {
          return '<p class="muted">Select an event in the flow canvas to inspect subsystem flow details, links, and payload context.</p>';
        }
        const eventId = node.event_id || node.node_id || '';
        const links = this.cognitiveFlow?.edges || [];
        const nodeMap = new Map((this.cognitiveFlow?.nodes || []).map(item => [item.event_id || item.node_id, item]));
        const incoming = links
          .filter(edge => edge.target_node_id === eventId)
          .map(edge => ({
            edge,
            sourceId: edge.source_node_id,
            sourceNode: nodeMap.get(edge.source_node_id),
          }))
          .slice(0, 10);
        const outgoing = links
          .filter(edge => edge.source_node_id === eventId)
          .map(edge => ({
            edge,
            targetId: edge.target_node_id,
            targetNode: nodeMap.get(edge.target_node_id),
          }))
          .slice(0, 10);
        const payload = node.event_payload || {};
        const candidates = this.cognitiveFlowAtlasNodeCandidates(node);
        const isDecisionSignal = Boolean(node.decision_signal);
        let html = '<div class="stack">';
        html += `<div class="pill-row">
          <span class="badge">${escapeHtml(node.subsystem || 'system')}</span>
          <span class="badge">${escapeHtml(node.kind || 'event')}</span>
          <span class="badge">${escapeHtml(node.pipeline_stage || 'processing')}</span>
          ${node.is_span_gate ? '<span class="badge warn">span gate</span>' : ''}
          ${isDecisionSignal ? '<span class="badge warn">decision signal</span>' : ''}
        </div>`;
        html += `<p class="muted">Time: ${formatDateTime(node.timestamp)} • Event: ${escapeHtml(eventId)}</p>`;
        html += `<p class="muted">Span: ${escapeHtml(node.span_id || '-')}${node.parent_span_id ? ` • Parent: ${escapeHtml(node.parent_span_id)}` : ''}</p>`;
        html += `<p class="muted">Source: ${escapeHtml(node.source_type || 'telemetry')} / ${escapeHtml(node.source_id || '-')} • ${escapeHtml(node.activation_source || '-')}</p>`;
        if (node.source_label) {
          html += `<p class="muted">Source label: ${escapeHtml(node.source_label)}</p>`;
        }
        if (node.label) {
          html += `<p><strong>${escapeHtml(node.label)}</strong></p>`;
        }
        if (node.message) {
          html += `<p class="muted">Message: ${escapeHtml(node.message)}</p>`;
        }
        if (node.event_message) {
          html += `<p class="muted">Event message: ${escapeHtml(node.event_message)}</p>`;
        }
        if (node.thought_text) {
          html += `<h5 class="mt-3">Observable thought</h5><p>${escapeHtml(node.thought_text)}</p>`;
        }
        if (node.reasoning_text) {
          html += `<h5 class="mt-3">Reasoning evidence</h5><p>${escapeHtml(node.reasoning_text)}</p>`;
        }
        if (node.evidence_text) {
          html += `<h5 class="mt-3">Referenced evidence</h5><p>${escapeHtml(node.evidence_text)}</p>`;
        }
        if (node.noted_as_such) {
          html += `<p class="muted">Marked as notable memory activity.</p>`;
        }
        if (candidates.length) {
          html += '<h5 class="mt-3">Linked memory nodes</h5><div class="pill-row">';
          candidates.forEach(candidate => {
            if (candidate.known) {
              html += `<button class="btn-link" onclick="window.__openCASMemoryApp.openAtlasNodeFromFlow('${escapeHtml(candidate.node_id)}')">${escapeHtml(candidate.node_id)} <small>[${escapeHtml(candidate.source)}]</small></button>`;
            } else {
              html += `<span class="badge">${escapeHtml(candidate.node_id)} <small>${escapeHtml(candidate.source)}</small></span>`;
            }
          });
          html += '</div>';
        }
        html += `<h5 class="mt-4">Payload</h5><pre class="json">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
        if (incoming.length || outgoing.length) {
          html += '<h5 class="mt-4">Connected flow edges</h5>';
          html += '<table class="data-table"><thead><tr><th>Direction</th><th>Connected Event</th><th>Kind</th><th>Strength</th><th>Action</th></tr></thead><tbody>';
          incoming.forEach(item => {
            const label = item.sourceNode?.label || item.sourceNode?.kind || item.sourceId || '-';
            html += `<tr>
              <td>in</td>
              <td>${escapeHtml(label)}</td>
              <td>${escapeHtml(item.edge.kind || '-')}</td>
              <td>${escapeHtml(String(item.edge.strength ?? '-'))}</td>
              <td><button class="btn-link" onclick="window.__openCASMemoryApp.selectCognitiveFlowNode('${escapeHtml(item.sourceId || '')}')">focus</button></td>
            </tr>`;
          });
          outgoing.forEach(item => {
            const label = item.targetNode?.label || item.targetNode?.kind || item.targetId || '-';
            html += `<tr>
              <td>out</td>
              <td>${escapeHtml(label)}</td>
              <td>${escapeHtml(item.edge.kind || '-')}</td>
              <td>${escapeHtml(String(item.edge.strength ?? '-'))}</td>
              <td><button class="btn-link" onclick="window.__openCASMemoryApp.selectCognitiveFlowNode('${escapeHtml(item.targetId || '')}')">focus</button></td>
            </tr>`;
          });
          html += '</tbody></table>';
        }
        html += '</div>';
        return html;
      },
      cognitiveFlowStripMarkup() {
        const nodes = this.cognitiveFlow?.nodes || [];
        if (!nodes.length) {
          return '<p class="muted">No live events to render yet.</p>';
        }
        const head = nodes.slice(-4).reverse();
        let html = '<div class="memory-activity-strip">';
        head.forEach(item => {
          const source = item.activation_source || item.kind || 'telemetry';
          const subsystem = item.flow_lane || item.subsystem || 'system';
          const stage = item.pipeline_stage || 'processing';
          const ts = formatDateTime(item.timestamp);
          html += `<span class="activity-chip">
            <span class="activity-dot" style="--pulse:50%"></span>
            <span>
              <strong>${escapeHtml(item.thought_text || item.label || item.node_id)}</strong>
              <small>${escapeHtml(source)} • ${escapeHtml(subsystem)}:${escapeHtml(stage)} • ${escapeHtml(ts || '')}</small>
            </span>
          </span>`;
        });
        html += '</div>';
        return html;
      },
      cognitiveFlowThoughtLedgerMarkup() {
        const stats = this.cognitiveFlow?.stats || {};
        let ledger = Array.isArray(stats.recent_readable_thoughts) ? stats.recent_readable_thoughts : [];
        if (!ledger.length) {
          ledger = (this.cognitiveFlow?.nodes || [])
            .filter(item => !item.is_noise && Number(item.flow_priority || 0) >= 0.25 && item.thought_text)
            .slice(-8);
        }
        if (!ledger.length) {
          return '<p class="muted">No readable thought records in this flow window. Increase the window or include more telemetry.</p>';
        }
        let html = '<div class="stack"><h4 class="mt-0">Readable thought ledger</h4>';
        html += '<p class="helper-text">High-signal cognitive events rendered as text. Use this when the canvas shape is not enough to understand what Bulma accessed, decided, or reasoned about.</p>';
        html += '<table class="data-table compact-table"><thead><tr><th>#</th><th>Lane</th><th>Thought / activity evidence</th><th>Reasoning</th><th></th></tr></thead><tbody>';
        ledger.slice(-8).forEach(item => {
          const eventId = item.event_id || item.node_id || '';
          html += `<tr>
            <td>${escapeHtml(String(item.sequence_index || item.temporal_order_label || '-'))}</td>
            <td>${escapeHtml(item.flow_lane || item.subsystem || 'system')}</td>
            <td><strong>${escapeHtml(item.label || item.kind || 'event')}</strong><br>${escapeHtml(item.thought_text || '')}${item.evidence_text ? `<br><small>${escapeHtml(item.evidence_text)}</small>` : ''}</td>
            <td>${escapeHtml(item.reasoning_text || (item.decision_signal ? 'decision signal' : ''))}</td>
            <td><button class="btn-link" onclick="window.__openCASMemoryApp.selectCognitiveFlowNode('${escapeHtml(eventId)}')">focus</button></td>
          </tr>`;
        });
        html += '</tbody></table></div>';
        return html;
      },
      renderCognitiveFlow() {
        const canvas = document.getElementById('memoryFlowCanvas');
        if (!canvas || this.cognitiveFlowMode !== 'flow') return;
        const rawNodes = (this.cognitiveFlow?.nodes || []).map(item => ({
          ...item,
          timestampMs: Date.parse(item.timestamp || ''),
        }));
        const nodes = rawNodes
          .filter(item => Number.isFinite(item.timestampMs))
          .sort((a, b) => a.timestampMs - b.timestampMs);
        const edges = (this.cognitiveFlow?.edges || []).slice();
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) {
          if (this.cognitiveFlowMode === 'flow') {
            setTimeout(() => this.scheduleCognitiveFlowRender(), 80);
          }
          return;
        }
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);
        if (!nodes.length) {
          this.cognitiveFlowNodePositions = [];
          ctx.fillStyle = '#94a3b8';
          ctx.fillText('No cognitive events in this window', 12, 24);
          return;
        }
        const nodeById = new Map(nodes.map(node => [node.event_id || node.node_id, node]));
        const times = nodes.map(node => Date.parse(node.timestamp || ''));
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        const spreadMs = Math.max(1, maxTime - minTime);
        const lanes = new Map();
        nodes.forEach((node) => {
          const lane = node.flow_lane || node.lane_label || node.subsystem || `Lane ${Number(node.span_depth || 0)}`;
          const bucket = lanes.get(lane) || [];
          bucket.push({ id: node.event_id || node.node_id, node });
          lanes.set(lane, bucket);
        });
        const sortedLanes = Array.from(lanes.keys()).sort((a, b) => {
          const first = lanes.get(a)?.[0]?.node;
          const second = lanes.get(b)?.[0]?.node;
          return Number(first?.lane_index ?? first?.span_depth ?? 0) - Number(second?.lane_index ?? second?.span_depth ?? 0);
        });
        const laneHeight = Math.max(40, Math.floor((rect.height - 100) / (sortedLanes.length || 1)));
        const plotLeft = 72;
        const plotRight = rect.width - 20;
        const plotTop = 24;
        const plotBottom = rect.height - 42;
        const plotHeight = Math.max(120, plotBottom - plotTop);
        const toX = ts => plotLeft + ((ts - minTime) / spreadMs) * (plotRight - plotLeft);
        const laneMap = new Map();
        sortedLanes.forEach((lane, index) => {
          laneMap.set(lane, plotBottom - (index + 0.5) * laneHeight);
        });
        const laneCounter = {};
        const positioned = nodes.map(node => {
          const lane = node.flow_lane || node.lane_label || node.subsystem || `Lane ${Number(node.span_depth || 0)}`;
          const baseY = laneMap.get(lane) || (plotBottom - 20);
          const serial = laneCounter[lane] ? laneCounter[lane] + 1 : 0;
          laneCounter[lane] = serial;
          const offset = ((serial % 4) - 1.5) * 10;
          const t = node.timestampMs;
          return {
            ...node,
            x: toX(t || minTime),
            y: baseY + offset,
          };
        });
        this.cognitiveFlowNodePositions = positioned;
        const positionedById = new Map(positioned.map(item => [item.event_id || item.node_id, item]));

        ctx.strokeStyle = 'rgba(0, 229, 255, 0.12)';
        ctx.lineWidth = 1;
        sortedLanes.forEach((lane, index) => {
          const y = laneMap.get(lane) || plotBottom;
          ctx.beginPath();
          ctx.moveTo(plotLeft, y);
          ctx.lineTo(plotRight, y);
          ctx.stroke();
          ctx.fillStyle = '#94a3b8';
          ctx.fillText(String(lane).substring(0, 24), 16, y + 4);
        });

        edges.forEach(edge => {
          if (!nodeById.get(edge.source_node_id) || !nodeById.get(edge.target_node_id)) return;
          const fromPos = positionedById.get(edge.source_node_id);
          const toPos = positionedById.get(edge.target_node_id);
          if (!fromPos || !toPos) return;
          const width = 1.2 + Math.max(0.4, Number(edge.strength || 0.35) * 2.1);
          ctx.strokeStyle = this.flowEdgeColor(edge.kind || '');
          ctx.lineWidth = width;
          const bx1 = fromPos.x;
          const by1 = fromPos.y;
          const bx2 = toPos.x;
          const by2 = toPos.y;
          ctx.beginPath();
          ctx.moveTo(bx1, by1);
          ctx.bezierCurveTo((bx1 + bx2) / 2, by1, (bx1 + bx2) / 2, by2, bx2, by2);
          ctx.stroke();
        });

        positioned.forEach((point) => {
          const r = this.flowRadius(point);
          ctx.beginPath();
          ctx.fillStyle = this.flowColor(point);
          const isSelected = this.cognitiveFlowSelectedNodeId === (point.event_id || point.node_id);
          ctx.strokeStyle = isSelected ? '#f8fafc' : (point.is_span_gate ? '#facc15' : 'rgba(8, 18, 31, 0.85)');
          ctx.lineWidth = isSelected ? 2.5 : (point.is_span_gate ? 2 : 1);
          ctx.arc(point.x, point.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          const shortLabel = point.label ? point.label.substring(0, 24) : point.kind || 'event';
          ctx.fillStyle = '#e2e8f0';
          ctx.fillText(shortLabel, point.x + 8, point.y - 8);
        });
        canvas.onclick = (evt) => {
          const rect = canvas.getBoundingClientRect();
          const x = evt.clientX - rect.left;
          const y = evt.clientY - rect.top;
          let best = null;
          let bestDist = Infinity;
          positioned.forEach(point => {
            const d = Math.hypot(point.x - x, point.y - y);
            if (d <= (this.flowRadius(point) + 8) && d < bestDist) {
              bestDist = d;
              best = point;
            }
          });
          if (best) {
            this.selectCognitiveFlowNode(best.event_id || best.node_id || '');
          }
        };
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
        const activityPlugin = {
          id: 'memoryActivityPulse',
          afterDatasetsDraw(chart) {
            const activityByNode = app.activityByNode();
            if (!activityByNode.size) return;
            const meta = chart.getDatasetMeta(0);
            const chartPoints = meta.data || [];
            const context = chart.ctx;
            const phase = (Date.now() % 1600) / 1600;
            context.save();
            nodes.forEach((node, index) => {
              const activity = activityByNode.get(node.node_id);
              if (!activity) return;
              const point = chartPoints[index];
              if (!point) return;
              const intensity = Math.max(0.1, Math.min(1, Number(activity.intensity || 0)));
              const base = Number(chart.data.datasets[0].pointRadius[index] || 6);
              const pulse = base + 8 + intensity * 14 + Math.sin(phase * Math.PI * 2) * 3;
              context.beginPath();
              context.strokeStyle = `rgba(34, 211, 238, ${0.25 + intensity * 0.55})`;
              context.lineWidth = 2 + intensity * 3;
              context.arc(point.x, point.y, pulse, 0, Math.PI * 2);
              context.stroke();
              context.beginPath();
              context.fillStyle = `rgba(34, 211, 238, ${0.08 + intensity * 0.12})`;
              context.arc(point.x, point.y, pulse * 0.72, 0, Math.PI * 2);
              context.fill();
            });
            context.restore();
          }
        };
        const style = this.chartNodeStyle(nodes);
        this.landscapeChart = new Chart(ctx, {
          type: 'scatter',
          plugins: [edgePlugin, haloPlugin, activityPlugin],
          data: {
            datasets: [{
              label: 'Memory atlas',
              data: nodes.map(node => ({ x: node.x, y: node.y })),
              backgroundColor: style.backgroundColor,
              pointRadius: style.pointRadius,
              pointBorderWidth: style.pointBorderWidth,
              pointBorderColor: style.pointBorderColor,
              pointStyle: style.pointStyle,
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
                    const activity = this.activityByNode().get(node.node_id);
                    if (activity) bits.push(`active ${Number(activity.intensity || 0).toFixed(2)}`);
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
        const proposalCount = Number(stats.visible_context_proposal_count || 0);
        const compactedCount = nodes.filter(node => node.compacted).length;
        const identityCoreCount = nodes.filter(node => node.identity_core).length;
        const salienceNodes = nodes.filter(node => node.salience !== null && node.salience !== undefined);
        const avgSalience = salienceNodes.length
          ? (salienceNodes.reduce((sum, node) => sum + Number(node.salience || 0), 0) / salienceNodes.length)
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
              <div><div class="stat-value">${proposalCount}</div><div class="stat-label">Proposals</div></div>
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
      memoryActivityMarkup() {
        const active = this.memoryActivity?.active_nodes || [];
        const events = this.memoryActivity?.events || [];
        const stats = this.memoryActivity?.stats || {};
        const live = active.length > 0;
        let html = `<div class="memory-activity-head">
          <div>
            <h4>Live Memory Activity</h4>
            <p class="muted">Nodes pulse when retrieval, prompt assembly, or memory maintenance activates them.</p>
          </div>
          <div class="pill-row">
            <span class="badge ${live ? 'ok' : 'badge-dim'}">${live ? 'active' : 'quiet'}</span>
            <span class="badge">${active.length} firing</span>
            <span class="badge">${events.length} events</span>
          </div>
        </div>`;
        if (!stats.available) {
          return html + '<p class="muted">Memory activity telemetry is not available for this runtime.</p>';
        }
        if (!active.length) {
          return html + '<p class="muted">No memory nodes have fired in the current live window.</p>';
        }
        html += '<div class="memory-activity-strip">';
        active.slice(0, 8).forEach(item => {
          const intensity = Math.round(Number(item.intensity || 0) * 100);
          const node = this.findNode(item.node_id);
          const label = node?.label || item.content_preview || item.node_id;
          const canFocus = Boolean(node);
          html += `<button class="activity-chip" ${canFocus ? `onclick="window.__openCASMemoryApp.selectNodeById('${escapeHtml(item.node_id)}')"` : ''}>
            <span class="activity-dot" style="--pulse:${Math.max(12, intensity)}%"></span>
            <span><strong>${escapeHtml(label)}</strong><small>${escapeHtml(item.activation_source || 'memory')} • ${item.event_count || 1} hits</small></span>
            <em>${intensity}%</em>
          </button>`;
        });
        html += '</div>';
        const recent = events.slice(0, 5).map(event => {
          const label = event.query || event.content_preview || event.node_id;
          return `<tr>
            <td>${escapeHtml(event.node_id)}</td>
            <td>${escapeHtml(event.activation_source || '-')}</td>
            <td>${escapeHtml(String(event.rank ?? '-'))}</td>
            <td>${escapeHtml(label || '')}</td>
          </tr>`;
        }).join('');
        html += `<table class="data-table compact-table mt-3"><thead><tr><th>Node</th><th>Source</th><th>Rank</th><th>Signal</th></tr></thead><tbody>${recent}</tbody></table>`;
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
        const salienceNodes = nodes.filter(n => n.salience !== null && n.salience !== undefined);
        const avgSalience = salienceNodes.length ? (salienceNodes.reduce((s, n) => s + Number(n.salience || 0), 0) / salienceNodes.length).toFixed(2) : '-';
        const kindDist = stats.kind_distribution || {};
        let html = `<div class="stat-grid">
          <div><div class="stat-value">${stats.visible_episode_count || 0}</div><div class="stat-label">Episodes</div></div>
          <div><div class="stat-value">${stats.visible_memory_count || 0}</div><div class="stat-label">Memories</div></div>
          <div><div class="stat-value">${stats.visible_context_proposal_count || 0}</div><div class="stat-label">Proposals</div></div>
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
            ${node.proposal_kind ? `<span class="badge">${escapeHtml(node.proposal_kind)}</span>` : ''}
            ${node.proposal_status ? `<span class="badge ${node.proposal_status === 'accepted' ? 'ok' : node.proposal_status === 'rejected' ? 'fail' : 'warn'}">${escapeHtml(node.proposal_status)}</span>` : ''}
            ${node.source_lane ? `<span class="badge">${escapeHtml(node.source_lane)}</span>` : ''}
            ${node.affect?.primary_emotion ? `<span class="badge">${escapeHtml(node.affect.primary_emotion)}</span>` : ''}
            ${node.identity_core ? '<span class="badge badge-gold">★ identity core</span>' : ''}
            ${node.compacted ? '<span class="badge badge-dim">compacted</span>' : ''}
          </div>
          ${node.session_id ? `<div class="mt-2"><button class="btn-link session-chip" onclick="window.__openCASMemoryApp.filterBySession('${escapeHtml(node.session_id)}')">${escapeHtml(node.session_id)}</button></div>` : ''}
          <p class="muted">Created: ${formatDateTime(node.created_at)} • Age: ${escapeHtml(String(node.age_days ?? '-'))}d</p>
          <p class="muted">Embedding: ${escapeHtml(node.embedding_model_id || 'none')} • Group: ${escapeHtml(node.projection_group || '-')}</p>
          <p class="muted">Salience: ${escapeHtml(String(node.salience ?? '-'))} • Confidence: ${escapeHtml(String(node.confidence_score ?? '-'))} • Connections: ${escapeHtml(String(node.connection_count ?? 0))}</p>
          ${node.authority ? `<p class="muted">Authority: ${escapeHtml(node.authority)} • Snapshot: ${escapeHtml(node.source_snapshot_id || '-')} • Epoch: ${escapeHtml(String(node.source_epoch ?? '-'))}</p>` : ''}
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
        if (node.node_type === 'context_proposal' && (node.evidence_refs || []).length) {
          html += `<h5 class="mt-4">Evidence Refs</h5><div class="pill-row">`;
          (node.evidence_refs || []).forEach(ref => {
            const raw = String(ref || '');
            const episodeNode = raw.startsWith('episode:') ? raw : `episode:${raw}`;
            const memoryNode = raw.startsWith('memory:') ? raw : `memory:${raw}`;
            const knownNode = this.findNode(raw) ? raw : this.findNode(episodeNode) ? episodeNode : this.findNode(memoryNode) ? memoryNode : '';
            if (knownNode) {
              html += `<button class="btn-link" onclick="window.__openCASMemoryApp.selectNodeById('${escapeHtml(knownNode)}')">${escapeHtml(raw)}</button>`;
            } else {
              html += `<span class="badge">${escapeHtml(raw)}</span>`;
            }
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
