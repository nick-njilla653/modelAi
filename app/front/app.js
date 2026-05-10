'use strict';

// ── Configuration ─────────────────────────────────────────────────────────────
const API = {
  BASE:      'http://localhost:8000',
  V1:        '/api/v1',
  QUERY:     '/api/v1/query',
  STREAM:    '/api/v1/query/stream',
  INGEST:    '/api/v1/ingest',
  HEALTH:    '/api/v1/health',
  EVAL_RUN:  '/api/v1/evaluation/run',
  EVAL_LIST: '/api/v1/evaluation/baselines',
};

// ── App state ─────────────────────────────────────────────────────────────────
const state = {
  sessionId:          null,
  streaming:          false,
  abortCtrl:          null,
  conversationHistory: [],   // [{q: string, a: string}]  — max 10 turns stored
  sessionTurnCount:   0,
};

// ── DOM shortcuts ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const $q = sel => document.querySelector(sel);

// ── Toast ──────────────────────────────────────────────────────────────────────
function toast(msg, type = 'info', ms = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// ── Navigation ─────────────────────────────────────────────────────────────────
function initNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const tab = item.dataset.tab;
      document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      item.classList.add('active');
      $(`tab-${tab}`).classList.add('active');
      if (tab === 'health') health.refresh();
    });
  });
}

// ── API helper ────────────────────────────────────────────────────────────────
async function apiCall(path, options = {}) {
  const url = API.BASE + path;
  const r = await fetch(url, options);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// ── MODULE : REQUÊTE ──────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
const query = {
  init() {
    $('btnSend').addEventListener('click', () => this.send());
    $('queryInput').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.send(); }
    });
    $('queryInput').addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 140) + 'px';
    });
    $('btnResetSession').addEventListener('click', () => {
      state.sessionId = null;
      state.conversationHistory = [];
      state.sessionTurnCount = 0;
      $('sessionBadge').style.display = 'none';
      toast('Nouvelle session démarrée', 'info');
    });
  },

  async send() {
    const text = $('queryInput').value.trim();
    if (!text || state.streaming) return;

    const lang     = $('qLang').value;
    const profile  = $('qProfile').value;
    const streaming= $('qStream').checked;

    // Remove welcome message
    const welcome = $('chatMessages').querySelector('.welcome-msg');
    if (welcome) welcome.remove();

    // Add user bubble
    appendMsg('user', text);
    $('queryInput').value = '';
    $('queryInput').style.height = 'auto';
    $('btnSend').disabled = true;

    if (streaming) {
      await this.sendStream(text, lang, profile);
    } else {
      await this.sendStandard(text, lang, profile);
    }

    $('btnSend').disabled = false;
  },

  async sendStandard(text, lang, profile) {
    const aiEl = appendMsg('ai', null); // placeholder
    const bubble = aiEl.querySelector('.msg-bubble');
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    const body = {
      query:           text,
      language:        lang,
      profile:         profile,
      session_id:      state.sessionId || undefined,
      session_context: buildSessionContext() || undefined,
    };

    try {
      const res = await apiCall(API.QUERY, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      });

      // Persist session id
      if (res.session_id) {
        state.sessionId = res.session_id;
        showSessionBadge(res.session_id);
      }

      // Record exchange in history
      recordExchange(text, res.answer);

      renderAIResponse(aiEl, res);
    } catch (err) {
      bubble.textContent = `Erreur : ${err.message}`;
      toast(err.message, 'error');
    }
  },

  async sendStream(text, lang, profile) {
    const aiEl = appendMsg('ai', null);
    const bubble = aiEl.querySelector('.msg-bubble');
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    const body = {
      query:           text,
      language:        lang,
      profile:         profile,
      session_id:      state.sessionId || undefined,
      session_context: buildSessionContext() || undefined,
    };

    state.streaming = true;
    state.abortCtrl = new AbortController();

    try {
      const r = await fetch(API.BASE + API.STREAM, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
        signal:  state.abortCtrl.signal,
      });

      if (!r.ok) throw new Error(`HTTP ${r.status}`);

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      let started  = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        // Parse SSE lines: "data: ..."
        chunk.split('\n').forEach(line => {
          if (!line.startsWith('data: ')) return;
          const raw = line.slice(6);

          // Marqueurs spéciaux (non JSON)
          if (raw.trim() === '[DONE]') return;
          if (raw.trim().startsWith('[ERROR]')) {
            bubble.textContent = raw.trim();
            return;
          }

          // Token JSON-encodé (préserve \n, espaces, accents)
          let token;
          try {
            token = JSON.parse(raw);
          } catch {
            token = raw; // fallback si payload non-JSON
          }

          if (!started) {
            bubble.textContent = '';
            started = true;
          }
          fullText += token;
          // Rendu Markdown progressif — re-render à chaque token contenant une nouvelle ligne
          if (token.includes('\n')) {
            bubble.classList.add('markdown');
            bubble.innerHTML = renderMarkdown(fullText);
          } else if (!bubble.classList.contains('markdown')) {
            bubble.textContent = fullText;
          } else {
            // Déjà en mode markdown : continuer à re-render pour rester cohérent
            bubble.innerHTML = renderMarkdown(fullText);
          }
          scrollChat();
        });
      }

      // Record exchange and update session display
      if (fullText) {
        if (!state.sessionId) {
          state.sessionId = 'stream-' + Date.now();
          showSessionBadge(state.sessionId);
        }
        recordExchange(text, fullText);
      }

      // Rendu Markdown complet à la fin du streaming
      if (fullText) {
        bubble.classList.add('markdown');
        bubble.innerHTML = renderMarkdown(fullText);
      }

      // Add minimal meta after streaming (no full QueryResponse available)
      const meta = document.createElement('div');
      meta.className = 'msg-meta';
      meta.innerHTML = `<span class="badge model"><i class="fas fa-robot"></i> streaming</span>`;
      aiEl.appendChild(meta);

    } catch (err) {
      if (err.name !== 'AbortError') {
        bubble.textContent = `Erreur : ${err.message}`;
        toast(err.message, 'error');
      }
    } finally {
      state.streaming = false;
      state.abortCtrl = null;
    }
  },
};

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderMarkdown(raw) {
  if (!raw) return '';

  // Échapper le HTML (sécurité XSS) avant d'ajouter les balises structurelles
  const esc = s => String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  // Formatage inline : gras, italique, code
  const inline = s => s
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>');

  const lines = raw.split('\n');
  let html = '';
  let inOl = false, inUl = false;

  const closeOl = () => { if (inOl) { html += '</ol>'; inOl = false; } };
  const closeUl = () => { if (inUl) { html += '</ul>'; inUl = false; } };
  const closeLists = () => { closeOl(); closeUl(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const e = esc(line);

    const mOl = line.match(/^(\d+)[.)]\s+(.+)/);
    const mUl = line.match(/^[-*]\s+(.+)/);
    const mH3 = line.match(/^###\s+(.+)/);
    const mH2 = line.match(/^##\s+(.+)/);
    const mH1 = line.match(/^#\s+(.+)/);

    if (mH1) { closeLists(); html += `<h2>${inline(esc(mH1[1]))}</h2>`; }
    else if (mH2) { closeLists(); html += `<h3>${inline(esc(mH2[1]))}</h3>`; }
    else if (mH3) { closeLists(); html += `<h4>${inline(esc(mH3[1]))}</h4>`; }
    else if (mOl) {
      closeUl();
      if (!inOl) { html += '<ol>'; inOl = true; }
      html += `<li>${inline(esc(mOl[2]))}</li>`;
    } else if (mUl) {
      closeOl();
      if (!inUl) { html += '<ul>'; inUl = true; }
      html += `<li>${inline(esc(mUl[1]))}</li>`;
    } else if (line.trim() === '') {
      closeLists();
      // Double newline → séparateur de paragraphe
      if (html && !html.endsWith('</p>') && !html.endsWith('<br>')) {
        html += '<br>';
      }
    } else {
      closeLists();
      html += `<p>${inline(e)}</p>`;
    }
  }
  closeLists();
  return html;
}

// ── Render AI response ────────────────────────────────────────────────────────
function renderAIResponse(aiEl, res) {
  const bubble = aiEl.querySelector('.msg-bubble');
  bubble.classList.add('markdown');
  bubble.innerHTML = renderMarkdown(res.answer);

  // Meta badges
  const meta = document.createElement('div');
  meta.className = 'msg-meta';

  const confidence = res.confidence_level || confidenceFromScore(res.uncertainty_score);
  meta.innerHTML += `<span class="badge ${confidence}">${confidenceLabel(confidence)}</span>`;

  if (res.uncertainty_score !== undefined) {
    const pct = (res.uncertainty_score * 100).toFixed(0);
    meta.innerHTML += `<span class="badge latency">score ${pct}%</span>`;
  }
  if (res.language_detected) {
    meta.innerHTML += `<span class="badge lang">${res.language_detected.toUpperCase()}</span>`;
  }
  if (res.model_used) {
    meta.innerHTML += `<span class="badge model"><i class="fas fa-microchip"></i> ${res.model_used}</span>`;
  }
  if (res.latency_ms) {
    meta.innerHTML += `<span class="badge latency"><i class="fas fa-clock"></i> ${res.latency_ms.toFixed(0)} ms</span>`;
  }

  // Safety flags
  (res.safety_flags || []).forEach(flag => {
    meta.innerHTML += `<span class="badge flag"><i class="fas fa-triangle-exclamation"></i> ${flag}</span>`;
  });

  aiEl.appendChild(meta);

  // Warnings
  if (res.warnings && res.warnings.length) {
    const w = document.createElement('div');
    w.className = 'warnings-block';
    w.innerHTML = res.warnings.map(
      ww => `<p><i class="fas fa-circle-exclamation"></i> ${escHtml(ww)}</p>`
    ).join('');
    aiEl.appendChild(w);
  }

  // Citations
  if (res.citations && res.citations.length) {
    const cBlock = document.createElement('div');
    cBlock.className = 'citations-block';
    cBlock.innerHTML = `<h4><i class="fas fa-quote-left"></i> Citations (${res.citations.length})</h4>`;
    res.citations.forEach(c => {
      cBlock.innerHTML += `
        <div class="citation-item">
          <i class="fas fa-file-lines"></i>
          <div>
            <div><span class="citation-source">${escHtml(c.source || '—')}</span>
              ${c.page ? `<span class="citation-page"> · p. ${c.page}</span>` : ''}
              ${c.score !== undefined ? `<span class="citation-page"> · score ${(c.score*100).toFixed(0)}%</span>` : ''}
            </div>
            ${c.excerpt ? `<div class="citation-excerpt">"${escHtml(truncate(c.excerpt, 160))}"</div>` : ''}
          </div>
        </div>`;
    });
    aiEl.appendChild(cBlock);
  }

  // Retrieved chunks (collapsible)
  if (res.retrieved_chunks && res.retrieved_chunks.length) {
    const btn = document.createElement('button');
    btn.className = 'chunks-toggle';
    btn.textContent = `▶ Voir ${res.retrieved_chunks.length} chunk(s) récupéré(s)`;
    const chunksList = document.createElement('div');
    chunksList.className = 'chunks-list';
    chunksList.style.display = 'none';

    res.retrieved_chunks.forEach(ch => {
      const score = ch.final_score || ch.rerank_score || ch.rrf_score || 0;
      const scoreClass = score >= 0.7 ? 'score-hi' : score >= 0.4 ? 'score-mid' : 'score-lo';
      chunksList.innerHTML += `
        <div class="chunk-card">
          <div class="chunk-header">
            <span class="chunk-source">${escHtml(ch.source || '—')}</span>
            ${ch.page ? `<span class="chunk-score">p. ${ch.page}</span>` : ''}
            <span class="chunk-score ${scoreClass}">▲ ${(score * 100).toFixed(0)}%</span>
            ${ch.language ? `<span class="badge lang">${ch.language.toUpperCase()}</span>` : ''}
          </div>
          <div class="chunk-content">${escHtml(truncate(ch.content || '', 240))}</div>
          ${ch.dense_score !== undefined ? `
            <div class="chunk-score" style="margin-top:4px;font-size:.7rem">
              dense ${(ch.dense_score*100).toFixed(0)}% &nbsp;|&nbsp;
              bm25 ${((ch.sparse_score||0)*100).toFixed(0)}% &nbsp;|&nbsp;
              rrf ${(ch.rrf_score||0).toFixed(4)}
              ${ch.rerank_score !== undefined ? `&nbsp;|&nbsp; rerank ${(ch.rerank_score*100).toFixed(0)}%` : ''}
            </div>` : ''}
        </div>`;
    });

    btn.addEventListener('click', () => {
      const hidden = chunksList.style.display === 'none';
      chunksList.style.display = hidden ? 'flex' : 'none';
      btn.textContent = hidden
        ? `▼ Masquer les chunks`
        : `▶ Voir ${res.retrieved_chunks.length} chunk(s) récupéré(s)`;
    });

    aiEl.appendChild(btn);
    aiEl.appendChild(chunksList);
  }

  scrollChat();
}

// ── Helpers msg ───────────────────────────────────────────────────────────────
function appendMsg(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `msg ${role}`;
  const roleLabel = { user: 'Vous', ai: 'GOV-AI 2.0', system: 'Système' };
  wrapper.innerHTML = `
    <div class="msg-role">${roleLabel[role] || role}</div>
    <div class="msg-bubble">${text !== null ? escHtml(text) : ''}</div>`;
  $('chatMessages').appendChild(wrapper);
  scrollChat();
  return wrapper;
}

function scrollChat() {
  const msgs = $('chatMessages');
  msgs.scrollTop = msgs.scrollHeight;
}

function showSessionBadge(id) {
  state.sessionId = id;
  $('sessionIdDisplay').textContent = `Session : ${id.slice(0, 8)}…`;
  $('sessionBadge').style.display = 'flex';
  updateTurnCounter();
}

function updateTurnCounter() {
  const el = $('sessionTurns');
  if (!el) return;
  const n = state.sessionTurnCount;
  el.textContent = n > 0 ? `· ${n} tour${n > 1 ? 's' : ''}` : '';
}

/**
 * Enregistre un échange (question + réponse) dans l'historique de session.
 * Conserve les MAX_HISTORY derniers échanges pour éviter de dépasser la limite.
 */
function recordExchange(question, answer) {
  const MAX_HISTORY = 10;
  const MAX_ANSWER_LEN = 300; // tronquer les réponses longues pour le contexte

  state.conversationHistory.push({
    q: question.trim(),
    a: answer.trim().slice(0, MAX_ANSWER_LEN) + (answer.length > MAX_ANSWER_LEN ? '…' : ''),
  });

  // Ne conserver que les MAX_HISTORY derniers échanges
  if (state.conversationHistory.length > MAX_HISTORY) {
    state.conversationHistory = state.conversationHistory.slice(-MAX_HISTORY);
  }

  state.sessionTurnCount++;
  updateTurnCounter();
}

/**
 * Construit le champ session_context à partir des derniers échanges.
 * Envoie les 5 échanges les plus récents (≈ contexte pertinent sans surcharger le LLM).
 * Retourne une chaîne vide si aucun historique.
 */
function buildSessionContext() {
  const CONTEXT_TURNS = 5;
  const recent = state.conversationHistory.slice(-CONTEXT_TURNS);
  if (recent.length === 0) return '';

  return recent.map(
    (turn, i) => `Tour ${state.conversationHistory.length - recent.length + i + 1} :\nQ: ${turn.q}\nR: ${turn.a}`
  ).join('\n\n');
}

function confidenceFromScore(s) {
  if (s === undefined || s === null) return 'medium';
  if (s >= 0.8) return 'high';
  if (s >= 0.6) return 'medium';
  if (s >= 0.3) return 'low';
  return 'insufficient';
}

function confidenceLabel(c) {
  return { high: '✓ Confiance haute', medium: '~ Confiance moyenne',
           low: '⚠ Confiance faible', insufficient: '✗ Insuffisant' }[c] || c;
}

function escHtml(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + '…' : s;
}

// ─────────────────────────────────────────────────────────────────────────────
// ── MODULE : INGESTION ───────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
const ingest = {
  file: null,

  init() {
    const dropZone = $('dropZone');
    const fileInput = $('fileInput');

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => this.setFile(e.target.files[0]));

    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      this.setFile(e.dataTransfer.files[0]);
    });

    $('btnClearFile').addEventListener('click', () => this.clearFile());
    $('btnIngest').addEventListener('click', () => this.run());
  },

  setFile(f) {
    if (!f) return;
    const ext = f.name.split('.').pop().toLowerCase();
    if (!['pdf', 'txt', 'md'].includes(ext)) {
      toast('Extension non supportée (PDF, TXT, MD)', 'error');
      return;
    }
    this.file = f;
    $('fileName').textContent = f.name;
    $('dropZone').style.display  = 'none';
    $('filePreview').style.display = 'flex';
    $('btnIngest').disabled = false;
  },

  clearFile() {
    this.file = null;
    $('fileInput').value = '';
    $('dropZone').style.display  = '';
    $('filePreview').style.display = 'none';
    $('btnIngest').disabled = true;
  },

  async run() {
    if (!this.file) return;

    const entryId = 'log-' + Date.now();
    this.addLogEntry(entryId, this.file.name, 'pending', 'Ingestion en cours…');

    const fd = new FormData();
    fd.append('file',         this.file);
    fd.append('doc_type',     $('docType').value);
    fd.append('institution',  $('institution').value);
    fd.append('jurisdiction', $('jurisdiction').value);
    fd.append('force_ocr',   $('forceOcr').checked ? 'true' : 'false');

    $('btnIngest').disabled = true;

    try {
      const res = await fetch(API.BASE + API.INGEST, { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();

      this.updateLogEntry(entryId, 'ok',
        `${data.chunks_created} chunk(s) · lang=${data.language_detected || '?'} · OCR=${data.ocr_used ? 'oui' : 'non'}`,
        data.document_id
      );
      toast(`Document ingéré : ${data.chunks_created} chunks`, 'success');
      this.clearFile();
    } catch (err) {
      this.updateLogEntry(entryId, 'error', err.message);
      toast(`Erreur ingestion : ${err.message}`, 'error');
      $('btnIngest').disabled = false;
    }
  },

  addLogEntry(id, filename, status, detail) {
    const log = $('ingestLog');
    log.querySelector('.empty-hint')?.remove();
    const el = document.createElement('div');
    el.id = id;
    el.className = `log-entry ${status}`;
    el.innerHTML = `
      <div class="log-entry-title"><i class="fas fa-file-alt"></i> ${escHtml(filename)}</div>
      <div class="log-entry-meta">${escHtml(detail)}</div>`;
    log.prepend(el);
  },

  updateLogEntry(id, status, detail, docId) {
    const el = $(id);
    if (!el) return;
    el.className = `log-entry ${status}`;
    const meta = el.querySelector('.log-entry-meta');
    meta.textContent = detail + (docId ? ` · id: ${docId.slice(0,8)}…` : '');
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// ── MODULE : ÉVALUATION ───────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
const evalModule = {
  init() {
    $('btnRunEval').addEventListener('click', () => this.run());
  },

  async run() {
    const baseline = $('evalBaseline').value;
    const dataset  = $('evalDataset').value.trim();
    const kRaw     = $('evalK').value.split(',').map(s => parseInt(s.trim(), 10)).filter(Boolean);

    $('btnRunEval').disabled = true;
    $('evalSpinner').style.display = 'block';
    $('evalResults').innerHTML = '<h3><i class="fas fa-table"></i> Résultats</h3><div class="spinner"></div>';

    try {
      const data = await apiCall(API.EVAL_RUN, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          baseline_id:  baseline,
          dataset_path: dataset,
          k_values:     kRaw,
        }),
      });
      this.renderResults(data);
      toast(`Évaluation ${baseline} terminée`, 'success');
    } catch (err) {
      $('evalResults').innerHTML = `
        <h3><i class="fas fa-table"></i> Résultats</h3>
        <p class="empty-hint" style="color:var(--danger)">Erreur : ${escHtml(err.message)}</p>`;
      toast(`Erreur évaluation : ${err.message}`, 'error');
    } finally {
      $('btnRunEval').disabled = false;
      $('evalSpinner').style.display = 'none';
    }
  },

  renderResults(d) {
    const ret = d.retrieval || {};
    const gen = d.generation || {};
    const sys = d.system?.end_to_end || {};
    const constraints = d.constraints_met || {};

    const kVals = Object.keys(ret.precision_at_k || {});

    let html = `
      <h3><i class="fas fa-table"></i> ${escHtml(d.baseline_description || d.baseline_id)}</h3>
      <p style="font-size:.78rem;color:var(--gray-400);margin-bottom:12px">
        ${d.num_queries} requêtes · ${d.timestamp?.slice(0,19)?.replace('T',' ') || ''}
      </p>`;

    // Retrieval metrics
    html += `<div class="metrics-section"><h4>Retrieval</h4><div class="metrics-grid">`;
    html += `<div class="metric-card"><div class="metric-value">${fmtPct(ret.mrr)}</div><div class="metric-label">MRR</div></div>`;
    kVals.forEach(k => {
      html += `<div class="metric-card">
        <div class="metric-value">${fmtPct(ret.ndcg_at_k?.[k])}</div>
        <div class="metric-label">nDCG@${k}</div>
      </div>`;
    });
    kVals.forEach(k => {
      html += `<div class="metric-card">
        <div class="metric-value">${fmtPct(ret.hit_rate_at_k?.[k])}</div>
        <div class="metric-label">HR@${k}</div>
      </div>`;
    });
    html += `</div></div>`;

    // Generation metrics
    html += `<div class="metrics-section"><h4>Génération</h4><div class="metrics-grid">`;
    const genMetrics = [
      { val: gen.citation_precision,  label: 'Citation Prec.', target: '≥ 95%' },
      { val: gen.hallucination_rate,  label: 'Hallucination',  target: '≤ 5%'  },
      { val: gen.faithfulness,        label: 'Faithfulness',   target: '—'      },
      { val: gen.isb,                 label: 'ISB',            target: '≥ 85%'  },
    ];
    genMetrics.forEach(m => {
      html += `<div class="metric-card">
        <div class="metric-value">${fmtPct(m.val)}</div>
        <div class="metric-label">${m.label}</div>
        ${m.target !== '—' ? `<div class="metric-status">${m.target}</div>` : ''}
      </div>`;
    });
    html += `</div></div>`;

    // Latency
    html += `<div class="metrics-section"><h4>Latence (contraintes mémoire)</h4><div class="metrics-grid">`;
    [['p50', '≤ 5 000 ms'], ['p95', '≤ 15 000 ms'], ['p99', '—']].forEach(([p, target]) => {
      const val = sys[`${p}_ms`];
      html += `<div class="metric-card">
        <div class="metric-value">${val !== undefined ? val.toFixed(0) + ' ms' : '—'}</div>
        <div class="metric-label">${p.toUpperCase()}</div>
        ${target !== '—' ? `<div class="metric-status">${target}</div>` : ''}
      </div>`;
    });
    html += `</div></div>`;

    // Constraints
    html += `<div class="metrics-section"><h4>Contraintes du mémoire</h4><div class="constraints-list">`;
    Object.entries(constraints).forEach(([key, ok]) => {
      html += `<div class="constraint-row">
        <i class="fas ${ok ? 'fa-circle-check ok' : 'fa-circle-xmark nok'}"></i>
        <span>${escHtml(key.replace(/_/g,' '))}</span>
        <span style="margin-left:auto;font-weight:600;color:var(--${ok?'success':'danger'})">${ok ? 'OK' : 'NON RESPECTÉE'}</span>
      </div>`;
    });
    html += `</div></div>`;

    $('evalResults').innerHTML = html;
  },
};

function fmtPct(v) {
  if (v === undefined || v === null) return '—';
  return (v * 100).toFixed(1) + '%';
}

// ─────────────────────────────────────────────────────────────────────────────
// ── MODULE : SANTÉ ────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
const HEALTH_ICONS = {
  milvus:        'fas fa-database',
  elasticsearch: 'fas fa-magnifying-glass',
  postgres:      'fas fa-server',
  embedding:     'fas fa-brain',
  api:           'fas fa-plug',
};

const health = {
  init() {
    $('btnRefreshHealth').addEventListener('click', () => this.refresh());
    this.refresh();
  },

  async refresh() {
    $('healthGrid').innerHTML = `<div class="health-card loading"><div class="spinner-sm"></div><span>Chargement…</span></div>`;
    $('healthRaw').textContent = '…';
    $('sidebarHealthDot').className = 'health-dot';
    $('sidebarHealthLabel').textContent = 'Vérification…';

    try {
      const data = await apiCall(API.HEALTH);
      this.render(data);
      const ok = data.status === 'ok';
      $('sidebarHealthDot').className = `health-dot ${ok ? 'ok' : 'degraded'}`;
      $('sidebarHealthLabel').textContent = ok ? 'Opérationnel' : 'Dégradé';
    } catch (err) {
      $('healthGrid').innerHTML = `<div class="health-card error">
        <div class="health-card-icon"><i class="fas fa-triangle-exclamation"></i></div>
        <div class="health-card-name">Backend</div>
        <div class="health-card-status">Inaccessible</div>
      </div>`;
      $('healthRaw').textContent = err.message;
      $('sidebarHealthDot').className = 'health-dot error';
      $('sidebarHealthLabel').textContent = 'Hors ligne';
    }
  },

  render(data) {
    $('healthRaw').textContent = JSON.stringify(data, null, 2);

    // API global
    const services = data.services || {};
    services['api'] = data.status || 'ok';

    const grid = $('healthGrid');
    grid.innerHTML = '';

    Object.entries(services).forEach(([name, status]) => {
      const isOk = status === 'ok';
      const isDeg = status === 'degraded' || String(status).startsWith('error');
      const cls = isOk ? 'ok' : isDeg ? 'error' : 'degraded';
      const icon = HEALTH_ICONS[name] || 'fas fa-circle-nodes';

      grid.innerHTML += `
        <div class="health-card ${cls}">
          <div class="health-card-icon"><i class="${icon}"></i></div>
          <div class="health-card-name">${escHtml(name)}</div>
          <div class="health-card-status">${escHtml(status)}</div>
        </div>`;
    });

    // Version / model
    if (data.version || data.model) {
      grid.innerHTML += `
        <div class="health-card ok" style="grid-column:1/-1;flex-direction:row;gap:16px;justify-content:center">
          ${data.version ? `<span style="font-size:.8rem;color:var(--gray-600)">v${escHtml(data.version)}</span>` : ''}
          ${data.model   ? `<span style="font-size:.8rem;color:var(--gray-600)"><i class="fas fa-robot"></i> ${escHtml(data.model)}</span>` : ''}
        </div>`;
    }
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// ── Bootstrap ─────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  query.init();
  ingest.init();
  evalModule.init();
  health.init();

  // Adjust chat input row layout
  const bar = document.querySelector('.chat-input-bar');
  if (bar) {
    const row = document.createElement('div');
    row.className = 'input-row';
    const ta   = $('queryInput');
    const btn  = $('btnSend');
    row.appendChild(ta);
    row.appendChild(btn);
    bar.appendChild(row);
  }
});
