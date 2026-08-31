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
  sessionId:           null,
  streaming:           false,
  abortCtrl:           null,
  conversationHistory: [],   // [{q, a}] — max 10 turns (fallback si pas de DB)
  sessionTurnCount:    0,
  serverHistoryLoaded: false, // true dès que le serveur renvoie session_id (Sprint 3)
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
      state.serverHistoryLoaded = false;
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
        state.serverHistoryLoaded = true;
        showSessionBadge(res.session_id);
      }

      // Record exchange in local history (fallback for streaming)
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
          if (raw.trim() === '[DONE]') {
            // Si aucun token reçu, vider la bulle (état indéfini → message d'erreur générique)
            if (!started) {
              bubble.textContent = 'Aucune réponse reçue. Vérifiez que le modèle LLM est chargé dans Ollama.';
            }
            return;
          }
          if (raw.trim().startsWith('[ERROR]')) {
            const errMsg = raw.trim().replace(/^\[ERROR\]\s*/, '');
            bubble.textContent = `⚠ ${errMsg}`;
            bubble.style.color = 'var(--error, #dc2626)';
            toast(errMsg, 'error');
            started = true; // évite le message générique du [DONE]
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

      // Metadata post-streaming
      const sMeta = document.createElement('div');
      sMeta.className = 'msg-meta';
      sMeta.innerHTML = `
        <span class="badge model"><i class="fas fa-robot"></i> streaming</span>
        <span class="badge lang">${$('qLang').value.toUpperCase()}</span>
        <span class="badge latency"><i class="fas fa-bolt"></i> temps réel</span>`;
      aiEl.appendChild(sMeta);

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

  // ── Meta badges ────────────────────────────────────────────────────────────
  const meta = document.createElement('div');
  meta.className = 'msg-meta';

  // Confidence
  const confidence = res.confidence_level || confidenceFromScore(res.uncertainty_score);
  meta.innerHTML += `<span class="badge ${confidence}">${confidenceLabel(confidence)}</span>`;

  // Score %
  if (res.uncertainty_score !== undefined) {
    const pct = (res.uncertainty_score * 100).toFixed(0);
    meta.innerHTML += `<span class="badge latency">score ${pct}%</span>`;
  }

  // Language
  if (res.language_detected) {
    meta.innerHTML += `<span class="badge lang">${res.language_detected.toUpperCase()}</span>`;
  }

  // Intent
  if (res.intent_detected && res.intent_detected !== 'unknown') {
    meta.innerHTML += `<span class="badge intent"><i class="fas fa-bullseye"></i> ${escHtml(res.intent_detected)}</span>`;
  }

  // Action plan badges (Sprint 3)
  const plan = res.action_plan || [];
  if (plan.includes('KNOWLEDGE_GRAPH')) {
    meta.innerHTML += `<span class="badge badge-kg"><i class="fas fa-circle-nodes"></i> KG</span>`;
  }
  if (plan.includes('WEB_SEARCH')) {
    meta.innerHTML += `<span class="badge badge-web"><i class="fas fa-globe"></i> Web</span>`;
  }

  // Model
  if (res.model_used) {
    meta.innerHTML += `<span class="badge model"><i class="fas fa-microchip"></i> ${escHtml(res.model_used)}</span>`;
  }

  // Latency
  if (res.latency_ms) {
    meta.innerHTML += `<span class="badge latency"><i class="fas fa-clock"></i> ${res.latency_ms.toFixed(0)} ms</span>`;
  }

  // Safety flags
  (res.safety_flags || []).forEach(flag => {
    meta.innerHTML += `<span class="badge flag"><i class="fas fa-triangle-exclamation"></i> ${escHtml(flag)}</span>`;
  });

  aiEl.appendChild(meta);

  // ── Warnings ────────────────────────────────────────────────────────────────
  if (res.warnings && res.warnings.length) {
    const w = document.createElement('div');
    w.className = 'warnings-block';
    w.innerHTML = res.warnings.map(ww => {
      const isWeb = ww.toLowerCase().includes('web');
      const icon = isWeb ? 'fa-globe' : 'fa-circle-exclamation';
      const cls  = isWeb ? 'warning-web' : '';
      return `<p class="${cls}"><i class="fas ${icon}"></i> ${escHtml(ww)}</p>`;
    }).join('');
    aiEl.appendChild(w);
  }

  // Citations
  if (res.citations && res.citations.length) {
    const cBlock = document.createElement('div');
    cBlock.className = 'citations-block';
    cBlock.innerHTML = `<h4><i class="fas fa-quote-left"></i> Citations (${res.citations.length})</h4>`;
    res.citations.forEach(c => {
      const relScore = c.relevance_score ?? c.score;
      cBlock.innerHTML += `
        <div class="citation-item">
          <i class="fas fa-file-lines"></i>
          <div>
            <div>
              <span class="citation-source">${escHtml(c.doc_title || c.source || '—')}</span>
              ${c.article ? `<span class="citation-page"> · ${escHtml(c.article)}</span>` : ''}
              ${c.page ? `<span class="citation-page"> · p. ${c.page}</span>` : ''}
              ${c.jurisdiction ? `<span class="citation-page"> · ${escHtml(c.jurisdiction)}</span>` : ''}
              ${relScore !== undefined ? `<span class="citation-score"> · ${(relScore*100).toFixed(0)}%</span>` : ''}
            </div>
            ${c.excerpt ? `<div class="citation-excerpt">"${escHtml(truncate(c.excerpt, 200))}"</div>` : ''}
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
            <div class="chunk-scores">
              <span title="Similarité cosinus">🧠 ${(ch.dense_score*100).toFixed(0)}%</span>
              <span title="BM25">📝 ${((ch.sparse_score||0)*100).toFixed(0)}%</span>
              <span title="RRF fusion">🔀 ${(ch.rrf_score||0).toFixed(4)}</span>
              ${ch.rerank_score !== undefined ? `<span title="Cross-encoder reranker" class="rerank-score">⚡ ${(ch.rerank_score*100).toFixed(0)}%</span>` : ''}
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
  const shortId = id.slice(0, 8);
  const dbIcon = state.serverHistoryLoaded
    ? `<i class="fas fa-database" title="Historique persisté en base de données" style="color:var(--success);font-size:.65rem"></i>`
    : '';
  $('sessionIdDisplay').innerHTML = `${dbIcon} Session : ${shortId}…`;
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
  neo4j:         'fas fa-circle-nodes',
  reranker:      'fas fa-sort-amount-down',
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
// ── MODULE : FINE-TUNING ──────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
const ft = {
  datasetId: null,
  activeJobId: null,
  ws: null,
  charts: {},

  init() {
    // Upload zone
    const dz = $('ftDropZone');
    const fi = $('ftFileInput');
    dz.addEventListener('click', () => fi.click());
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => {
      e.preventDefault(); dz.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) this._setFile(f);
    });
    fi.addEventListener('change', () => { if (fi.files[0]) this._setFile(fi.files[0]); });
    $('ftClearFile').addEventListener('click', () => this._clearFile());
    $('btnFtUpload').addEventListener('click', () => this.upload());
    $('btnFtStart').addEventListener('click', () => this.startJob());
    $('btnFtApply').addEventListener('click', () => this.applyModel());
    $('btnFtEvaluate').addEventListener('click', () => this._openEvalSection());
    $('btnRollbackEmbedding').addEventListener('click', () => this.rollback('embedding'));
    $('btnRollbackLlm').addEventListener('click', () => this.rollback('llm'));
    $('btnRollbackReranker').addEventListener('click', () => this.rollback('reranker'));
    $('btnRefreshJobs').addEventListener('click', () => this.loadJobs());
    $('btnFtEvalStart').addEventListener('click', () => this.runEvaluation());
    this.loadJobs();
    this.loadActiveModels();
  },

  _setFile(f) {
    $('ftFileName').textContent = f.name;
    $('ftFilePreview').style.display = 'flex';
    $('ftDropZone').style.display = 'none';
    $('btnFtUpload').disabled = false;
    this._selectedFile = f;
  },

  _clearFile() {
    this._selectedFile = null;
    this.datasetId = null;
    $('ftFilePreview').style.display = 'none';
    $('ftDropZone').style.display = 'flex';
    $('btnFtUpload').disabled = true;
    $('btnFtStart').disabled = true;
    $('ftUploadResult').style.display = 'none';
    $('ftFileInput').value = '';
  },

  async upload() {
    if (!this._selectedFile) return;
    const btn = $('btnFtUpload');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Upload en cours…';
    try {
      const fd = new FormData();
      fd.append('file', this._selectedFile);
      const r = await fetch(API.BASE + '/api/v1/finetune/upload', { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || 'Erreur upload');
      const data = await r.json();
      this.datasetId = data.dataset_id;
      $('ftUploadResult').style.display = 'block';
      $('ftUploadResult').innerHTML = `
        <div class="ft-upload-stats">
          <span><i class="fas fa-file-alt"></i> ${data.supported_files} fichier(s) supporté(s)</span>
          <span><i class="fas fa-weight-hanging"></i> ${data.zip_size_mb} Mo</span>
          <span class="badge-ok"><i class="fas fa-check"></i> ${data.message}</span>
        </div>`;
      $('btnFtStart').disabled = false;
      toast(`Dossier analysé : ${data.supported_files} document(s) prêts`, 'success');
    } catch (err) {
      toast('Erreur upload : ' + err.message, 'error');
      btn.disabled = false;
    }
    btn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Analyser le dossier';
  },

  async startJob() {
    if (!this.datasetId) return;
    const target = $('ftTarget').value;
    const mode   = $('ftMode').value;
    const payload = {
      name:           $('ftJobName').value || 'Fine-Tuning GOV-AI',
      target,
      dataset_id:     this.datasetId,
      epochs:         parseInt($('ftEpochs').value) || 3,
      batch_size:     parseInt($('ftBatch').value) || 16,
      learning_rate:  parseFloat($('ftLr').value) || 2e-5,
      pairs_per_chunk:parseInt($('ftPairsPerChunk').value) || 3,
      lora_mode:      mode === 'lora',
      ingest_corpus:  $('ftIngestCorpus').checked,
    };
    try {
      const r = await fetch(API.BASE + '/api/v1/finetune/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error((await r.json()).detail || 'Erreur');
      const data = await r.json();
      this.activeJobId = data.job_id;
      this._showProgressCard(data.job_id);
      this._connectWs(data.job_id);
      this.loadJobs();
      toast('Entraînement lancé !', 'success');
    } catch (err) {
      toast('Erreur : ' + err.message, 'error');
    }
  },

  _showProgressCard(jobId) {
    const card = $('ftProgressCard');
    card.style.display = 'block';
    $('ftProgressBar').style.width = '0%';
    $('ftProgressPct').textContent = '0%';
    $('ftProgressLog').innerHTML = '';
    $('ftProgressActions').style.display = 'none';
    // Réinitialiser le rapport leakage et les flags d'affichage unique
    const lkReport = $('ftLeakageReport');
    if (lkReport) lkReport.style.display = 'none';
    this._leakageShown = false;
    this._corpusShown  = false;
    // Réinitialiser le retry WebSocket et le polling éventuel
    this._wsRetries = 0;
    this._wsJobId   = null;
    if (this._pollInterval) { clearInterval(this._pollInterval); this._pollInterval = null; }

    // Afficher les boutons de contrôle et câbler les handlers
    const controls = $('ftJobControls');
    if (controls) {
      controls.style.display = 'flex';
      $('btnFtPause').style.display  = '';
      $('btnFtResume').style.display = 'none';

      $('btnFtCancel').onclick = async () => {
        if (!confirm('Annuler le fine-tuning en cours ? Cette action est irréversible.')) return;
        try {
          await fetch(`${API.BASE}${API.V1}/finetune/jobs/${jobId}/cancel`, { method: 'POST' });
          toast('Annulation demandée…', 'warning');
        } catch { toast('Impossible d\'annuler (réseau)', 'error'); }
      };
      $('btnFtPause').onclick = async () => {
        try {
          await fetch(`${API.BASE}${API.V1}/finetune/jobs/${jobId}/pause`, { method: 'POST' });
          toast('Mise en pause…', 'info');
        } catch { toast('Impossible de mettre en pause', 'error'); }
      };
      $('btnFtResume').onclick = async () => {
        try {
          await fetch(`${API.BASE}${API.V1}/finetune/jobs/${jobId}/resume`, { method: 'POST' });
          toast('Reprise en cours…', 'info');
        } catch { toast('Impossible de reprendre', 'error'); }
      };
    }

    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  _connectWs(jobId) {
    // Fermeture propre du WS précédent (évite "closed before established")
    if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
      this.ws.onclose = null; // désactive le retry de l'ancien WS
      this.ws.close();
    }

    this._wsJobId   = jobId;
    this._wsRetries = this._wsRetries || 0;

    // URL dérivée de window.location → pas de cross-origin, Edge Tracking OK
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${proto}://${window.location.host}/api/v1/finetune/progress/${jobId}`;

    try {
      this.ws = new WebSocket(wsUrl);
    } catch (e) {
      this._startPolling(jobId);
      return;
    }

    this.ws.onopen = () => {
      this._wsRetries = 0;
      // Annuler le polling si on avait basculé dessus
      if (this._pollInterval) { clearInterval(this._pollInterval); this._pollInterval = null; }
    };

    this.ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.ping) return;
        this._updateProgress(data);
      } catch { /* ignore JSON malformé */ }
    };

    this.ws.onerror = () => { /* géré dans onclose */ };

    this.ws.onclose = () => {
      if (this._wsJobId !== jobId) return; // job changé, ne pas retenter
      if (this._wsRetries < 5) {
        this._wsRetries++;
        const delay = Math.min(500 * Math.pow(2, this._wsRetries), 15000);
        setTimeout(() => this._connectWs(jobId), delay);
      } else {
        // 5 échecs → fallback polling REST
        this._startPolling(jobId);
      }
    };
  },

  _startPolling(jobId) {
    if (this._pollInterval) clearInterval(this._pollInterval);
    toast('WebSocket indisponible — suivi en mode polling (3 s)', 'warning');
    this._pollInterval = setInterval(async () => {
      try {
        const r = await fetch(`${API.BASE}${API.V1}/finetune/jobs/${jobId}`);
        if (!r.ok) return;
        const data = await r.json();
        this._updateProgress(data);
        if (['completed', 'failed', 'cancelled'].includes(data.status)) {
          clearInterval(this._pollInterval);
          this._pollInterval = null;
        }
      } catch { /* réseau temporairement indisponible */ }
    }, 3000);
  },

  _updateProgress(job) {
    const pct = Math.round(job.progress || 0);
    $('ftProgressBar').style.width = pct + '%';
    $('ftProgressPct').textContent = pct + '%';

    const logs = job.logs || [];
    if (logs.length) {
      $('ftProgressLog').innerHTML = logs.slice(-6).map(l =>
        `<div class="ft-log-line">${escHtml(l)}</div>`
      ).join('');
    }

    // Affichage du statut d'ingestion corpus
    if (job.corpus_ingestion && !this._corpusShown) {
      this._corpusShown = true;
      const ci = job.corpus_ingestion;
      if (ci.error) {
        toast(`Ingestion corpus partielle : ${ci.error}`, 'warning');
      } else {
        toast(`Corpus enrichi : ${ci.ingested} doc(s) → KG + Milvus + ES`, 'success');
      }
    }

    // ── Rapport data leakage (affiché une seule fois quand il arrive) ────
    if (job.leakage_report && !this._leakageShown) {
      this._leakageShown = true;
      this._renderLeakageReport(job.leakage_report, job.leakage_details || []);
    }

    // Affichage de la re-indexation Milvus (après apply embedding)
    if (job._reindex_update) {
      const reindexPct = job.reindex_progress || 0;
      const reindexLog = job.reindex_log || '';
      if (reindexPct < 100) {
        toast(`Re-indexation Milvus : ${reindexPct}% — ${reindexLog}`, 'info', 4000);
      } else if (job.reindex_status === 'completed') {
        toast(`Milvus re-indexé : ${job.reindex_result?.reindexed || 0} vecteurs recalculés`, 'success');
      }
    }

    // ── Boutons Cancel / Pause / Resume ──────────────────────────────────
    const controls  = $('ftJobControls');
    const pauseBtn  = $('btnFtPause');
    const resumeBtn = $('btnFtResume');
    const cancelBtn = $('btnFtCancel');
    const activeStatuses = ['running', 'dataset_building', 'leakage_check', 'ingesting_corpus', 'training', 'cancelling'];

    if (controls) {
      const isActive = activeStatuses.includes(job.status);
      const isPaused = job.status === 'paused';

      controls.style.display = (isActive || isPaused) ? 'flex' : 'none';
      if (pauseBtn)  pauseBtn.style.display  = isPaused ? 'none' : '';
      if (resumeBtn) resumeBtn.style.display = isPaused ? '' : 'none';
      if (cancelBtn) cancelBtn.disabled = job.status === 'cancelling';
    }

    if (job.status === 'completed') {
      $('ftProgressCard').querySelector('h3').innerHTML =
        '<i class="fas fa-check-circle" style="color:var(--success)"></i> Entraînement terminé !';
      $('ftProgressActions').style.display = 'flex';
      if (controls) controls.style.display = 'none';
      if (job.result?.loss_history?.length) this._drawLossCurve(job.result.loss_history);
      this.loadJobs();
      this.loadActiveModels();
    } else if (job.status === 'cancelled') {
      $('ftProgressCard').querySelector('h3').innerHTML =
        '<i class="fas fa-ban" style="color:var(--warning,#f59e0b)"></i> Fine-tuning annulé';
      if (controls) controls.style.display = 'none';
      if (this._pollInterval) { clearInterval(this._pollInterval); this._pollInterval = null; }
      this._wsJobId = null;
      toast('Fine-tuning annulé.', 'warning');
    } else if (job.status === 'failed') {
      $('ftProgressCard').querySelector('h3').innerHTML =
        '<i class="fas fa-times-circle" style="color:var(--error)"></i> Échec du fine-tuning';
      if (controls) controls.style.display = 'none';
      toast('Erreur fine-tuning : ' + (job.error || 'inconnue'), 'error');
    } else if (job.status === 'paused') {
      $('ftProgressCard').querySelector('h3').innerHTML =
        '<i class="fas fa-pause-circle" style="color:var(--primary)"></i> En pause — en attente de reprise';
    }
  },

  _renderLeakageReport(summary, details) {
    const el = $('ftLeakageReport');
    if (!el) return;

    const leaked  = summary.leaked  || 0;
    const clean   = summary.clean   || 0;
    const pct     = summary.leakage_pct || 0;
    const levels  = (summary.levels_triggered || []).join(' · ') || 'aucun';
    const warning = summary.warning || '';

    // Sévérité
    let severity = 'ok', icon = 'fa-shield-check', label = 'Aucun leakage';
    if (warning) {
      severity = 'warning'; icon = 'fa-triangle-exclamation'; label = 'Vérification ignorée';
    } else if (pct > 30) {
      severity = 'danger'; icon = 'fa-radiation'; label = `Leakage élevé (${pct}%)`;
    } else if (pct > 0) {
      severity = 'warning'; icon = 'fa-triangle-exclamation'; label = `Leakage partiel (${pct}%)`;
    }

    $('ftLeakageBadge').className = `ft-leakage-badge ${severity}`;
    $('ftLeakageBadge').innerHTML = `<i class="fas ${icon}"></i> ${label}`;
    $('ftLeakageSummaryText').textContent =
      warning
        ? warning
        : `${leaked} paire(s) exclue(s) sur ${summary.total_train} · ${clean} propres · Niveaux : ${levels}`;

    // Détails par question d'évaluation
    const detailsEl = $('ftLeakageDetails');
    if (!details.length) {
      detailsEl.innerHTML = '<div class="ft-leakage-empty">Aucune question d\'évaluation affectée.</div>';
    } else {
      detailsEl.innerHTML = details.map(row => {
        const l1 = row.level_1_source_matches   || 0;
        const l2 = row.level_2_keyword_matches  || 0;
        const l3 = row.level_3_semantic_matches || 0;
        const lvlBadges = [
          l1 ? `<span class="ft-leakage-lvl l1">L1 source ×${l1}</span>` : '',
          l2 ? `<span class="ft-leakage-lvl l2">L2 mots-clés ×${l2}</span>` : '',
          l3 ? `<span class="ft-leakage-lvl l3">L3 sémantique ×${l3}</span>` : '',
        ].filter(Boolean).join(' ');
        return `<div class="ft-leakage-row">
          <span class="lk-query">[${escHtml(row.eval_query_id)}] ${escHtml(row.eval_query)}</span>
          <span style="display:flex;gap:4px;flex-wrap:wrap">${lvlBadges}</span>
        </div>`;
      }).join('');
    }

    el.style.display = 'block';
    if (pct > 0) toast(`Data leakage : ${leaked} paire(s) filtrée(s) (${pct}%)`, pct > 30 ? 'error' : 'warning');
  },

  async applyModel() {
    if (!this.activeJobId) return;
    try {
      const r = await fetch(API.BASE + `/api/v1/finetune/apply/${this.activeJobId}`, { method: 'POST' });
      if (!r.ok) throw new Error((await r.json()).detail || 'Erreur');
      const data = await r.json();
      toast(`Modèle intégré : ${data.applied}`, 'success');
      this.loadActiveModels();
    } catch (err) {
      toast('Erreur intégration : ' + err.message, 'error');
    }
  },

  async rollback(type) {
    try {
      const r = await fetch(API.BASE + `/api/v1/finetune/rollback/${type}`, { method: 'POST' });
      if (!r.ok) throw new Error((await r.json()).detail || 'Aucun modèle précédent');
      toast(`Rollback ${type} effectué`, 'info');
      this.loadActiveModels();
    } catch (err) {
      toast(err.message, 'warning');
    }
  },

  async loadJobs() {
    try {
      const data = await apiCall('/api/v1/finetune/jobs');
      this._renderJobList(data.jobs || []);
      this._populateEvalSelector(data.jobs || []);
    } catch (err) {
      $('ftJobList').innerHTML = '<p class="empty-hint">Erreur de chargement.</p>';
    }
  },

  async loadActiveModels() {
    try {
      const data = await apiCall('/api/v1/finetune/active-models');
      const row = (icon, label, val, isFt) => `
        <div class="ft-model-row">
          <span class="ft-model-label"><i class="fas fa-${icon}"></i> ${label}</span>
          <span class="ft-model-val">${escHtml((val || '—').split('/').pop())}</span>
          ${isFt ? '<span class="badge-ft">Fine-tuné</span>' : '<span class="badge-orig">Original</span>'}
        </div>`;
      $('ftActiveModels').innerHTML =
        row('robot',         'LLM',       data.llm_model?.ollama_model_name,       !!data.llm_model?.job_id) +
        row('vector-square', 'Embedding', data.embedding_model?.model_path,        !!data.embedding_model?.job_id) +
        row('filter',        'Reranker',  data.reranker_model?.model_path,         !!data.reranker_model?.job_id);
    } catch(e) {
      $('ftActiveModels').innerHTML = '<p class="empty-hint">Erreur</p>';
    }
  },

  _renderJobList(jobs) {
    if (!jobs.length) {
      $('ftJobList').innerHTML = '<p class="empty-hint">Aucun job pour l\'instant.</p>';
      return;
    }
    $('ftJobList').innerHTML = jobs.map(j => `
      <div class="ft-job-row ${j.status}">
        <div class="ft-job-info">
          <strong>${escHtml(j.name)}</strong>
          <span class="ft-job-meta">
            <i class="fas fa-tag"></i> ${j.target}
            &nbsp;·&nbsp; <i class="fas fa-${_statusIcon(j.status)}"></i> ${j.status}
            &nbsp;·&nbsp; ${Math.round(j.progress || 0)}%
          </span>
        </div>
        <div class="ft-job-btns">
          ${j.status === 'completed' ? `
            <button class="btn-sm" onclick="ft._resumeJob('${j.id}', '${j.target}')">
              <i class="fas fa-eye"></i> Voir
            </button>` : ''}
        </div>
      </div>`).join('');
  },

  _resumeJob(jobId, _target) {
    this.activeJobId = jobId;
    $('ftProgressCard').style.display = 'block';
    $('ftProgressBar').style.width = '100%';
    $('ftProgressPct').textContent = '100%';
    $('ftProgressCard').querySelector('h3').innerHTML =
      '<i class="fas fa-check-circle" style="color:var(--success)"></i> Job complété';
    $('ftProgressActions').style.display = 'flex';
  },

  _populateEvalSelector(jobs) {
    const sel = $('ftEvalJobSelect');
    const completed = jobs.filter(j => j.status === 'completed');
    sel.innerHTML = completed.length
      ? completed.map(j => `<option value="${j.id}">${escHtml(j.name)} (${j.target})</option>`).join('')
      : '<option value="">Aucun job complété</option>';
    $('ftEvalCard').style.display = completed.length ? 'block' : 'none';
  },

  _openEvalSection() {
    $('ftEvalCard').scrollIntoView({ behavior: 'smooth' });
  },

  async runEvaluation() {
    const jobId = $('ftEvalJobSelect').value;
    if (!jobId) return;
    $('btnFtEvalStart').disabled = true;
    $('ftEvalSpinner').style.display = 'block';
    $('ftMetricsSummary').style.display = 'none';
    $('ftChartsContainer').style.display = 'none';
    $('ftSampleResponses').style.display = 'none';

    try {
      // Lance l'évaluation
      await fetch(API.BASE + `/api/v1/finetune/evaluate/${jobId}`, { method: 'POST' });
      toast('Évaluation lancée, patientez…', 'info');

      // Poll résultats toutes les 10s (max 3 min)
      let attempts = 0;
      const poll = setInterval(async () => {
        attempts++;
        try {
          const r = await fetch(API.BASE + `/api/v1/finetune/evaluate/${jobId}/results`);
          if (r.ok) {
            clearInterval(poll);
            const data = await r.json();
            this._renderEvalResults(data);
          } else if (attempts > 18) {
            clearInterval(poll);
            toast('Timeout évaluation', 'warning');
          }
        } catch(e) {}
      }, 10000);

    } catch (err) {
      toast('Erreur : ' + err.message, 'error');
    }
    $('ftEvalSpinner').style.display = 'none';
    $('btnFtEvalStart').disabled = false;
  },

  _renderEvalResults(data) {
    const orig = data.original_metrics || {};
    const ft   = data.finetuned_metrics || {};
    const refs = data.reference_benchmarks || {};

    // Métriques résumé
    $('ftMetricsSummary').style.display = 'grid';
    $('ftMetricsSummary').innerHTML = [
      { label: 'BLEU (original)',        val: orig.mean_bleu        || 0 },
      { label: 'BLEU (fine-tuné)',        val: ft.mean_bleu          || 0 },
      { label: 'ROUGE-L (original)',      val: orig.mean_rouge_l     || 0 },
      { label: 'ROUGE-L (fine-tuné)',     val: ft.mean_rouge_l       || 0 },
      { label: 'Score juge (original)',   val: orig.mean_judge_score || 0 },
      { label: 'Score juge (fine-tuné)', val: ft.mean_judge_score   || 0 },
      { label: 'Win rate fine-tuné',      val: ft.win_rate           || 0 },
      { label: 'Latence ms (fine-tuné)', val: ft.mean_latency_ms    || 0 },
    ].map(m => `
      <div class="ft-metric-card">
        <div class="ft-metric-val">${typeof m.val === 'number' && m.val < 1 && m.val > 0 ? (m.val*100).toFixed(1)+'%' : Number(m.val).toFixed(m.label.includes('ms') ? 0 : 3)}</div>
        <div class="ft-metric-label">${m.label}</div>
      </div>`).join('');

    $('ftChartsContainer').style.display = 'block';

    // Bar chart BLEU + ROUGE-L
    this._drawBarChart(orig, ft);

    // Radar chart comparaison mondiale
    this._drawRadarChart(orig, ft, refs, data.finetuned_model, data.original_model);

    // Exemples de réponses
    if (data.sample_responses?.length) {
      this._renderSamples(data.sample_responses);
    }
  },

  _drawBarChart(orig, ft) {
    const ctx = $('chartBarMetrics').getContext('2d');
    if (this.charts.bar) this.charts.bar.destroy();
    this.charts.bar = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['BLEU', 'ROUGE-L', 'Score juge /10', 'Win rate'],
        datasets: [
          {
            label: 'Modèle original',
            backgroundColor: '#2563eb99',
            data: [
              (orig.mean_bleu || 0)*100,
              (orig.mean_rouge_l || 0)*100,
              (orig.mean_judge_score || 0)*10,
              (orig.win_rate || 0)*100,
            ],
          },
          {
            label: 'Modèle fine-tuné',
            backgroundColor: '#16a34a99',
            data: [
              (ft.mean_bleu || 0)*100,
              (ft.mean_rouge_l || 0)*100,
              (ft.mean_judge_score || 0)*10,
              (ft.win_rate || 0)*100,
            ],
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'top' }, title: { display: true, text: 'Métriques BLEU / ROUGE-L / Juge (%)' } },
        scales: { y: { min: 0, max: 100 } },
      },
    });
  },

  _drawRadarChart(orig, ft, refs, ftModelName, origModelName) {
    const ctx = $('chartRadarComparison').getContext('2d');
    if (this.charts.radar) this.charts.radar.destroy();

    const labels = ['BLEU', 'ROUGE-L', 'Score juge', 'Win rate', 'Vitesse'];
    const normalize = (v, max) => Math.min(100, (v / max) * 100);

    const datasets = [
      {
        label: origModelName || 'Original',
        borderColor: '#2563eb',
        backgroundColor: '#2563eb22',
        data: [
          normalize(orig.mean_bleu || 0, 1),
          normalize(orig.mean_rouge_l || 0, 1),
          normalize(orig.mean_judge_score || 0, 10),
          normalize(orig.win_rate || 0, 1),
          normalize(1000 / Math.max(orig.mean_latency_ms || 1000, 1), 1),
        ],
      },
      {
        label: ftModelName || 'Fine-tuné',
        borderColor: '#16a34a',
        backgroundColor: '#16a34a22',
        data: [
          normalize(ft.mean_bleu || 0, 1),
          normalize(ft.mean_rouge_l || 0, 1),
          normalize(ft.mean_judge_score || 0, 10),
          normalize(ft.win_rate || 0, 1),
          normalize(1000 / Math.max(ft.mean_latency_ms || 1000, 1), 1),
        ],
      },
    ];

    // Ajouter les modèles de référence
    const refColors = { 'gpt-4o': '#10a37f', 'claude-3-5-sonnet': '#d97706', 'mistral-7b': '#7c3aed', 'llama3.2-3b': '#94a3b8' };
    Object.entries(refs).forEach(([key, ref]) => {
      datasets.push({
        label: ref.label,
        borderColor: ref.color || refColors[key] || '#94a3b8',
        backgroundColor: (ref.color || '#94a3b8') + '11',
        borderDash: [5, 5],
        data: [
          normalize(ref.mteb_score || 0, 1) * 0.8,
          normalize(ref.rag_faithfulness || 0, 1) * 85,
          normalize(ref.rag_answer_relevancy || 0, 1) * 10 / 1,
          normalize(ref.rag_faithfulness || 0, 1) * 65,
          60,
        ],
      });
    });

    this.charts.radar = new Chart(ctx, {
      type: 'radar',
      data: { labels, datasets },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'bottom', labels: { font: { size: 11 } } },
          title: { display: true, text: 'Comparaison mondiale des modèles (valeurs normalisées)' },
        },
        scales: { r: { min: 0, max: 100, ticks: { stepSize: 20 } } },
      },
    });
  },

  _drawLossCurve(history) {
    $('chartLossWrap').style.display = 'block';
    const ctx = $('chartLossCurve').getContext('2d');
    if (this.charts.loss) this.charts.loss.destroy();
    this.charts.loss = new Chart(ctx, {
      type: 'line',
      data: {
        labels: history.map((_, i) => `Step ${i+1}`),
        datasets: [{
          label: 'Training loss',
          data: history.map(h => h.loss ?? h.score),
          borderColor: '#2563eb',
          backgroundColor: '#2563eb22',
          tension: 0.3,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        plugins: { title: { display: true, text: 'Courbe de perte (Training Loss)' } },
        scales: { y: { title: { display: true, text: 'Loss' } } },
      },
    });
  },

  _renderSamples(samples) {
    $('ftSampleResponses').style.display = 'block';
    $('ftSampleResponses').innerHTML = `
      <h4 style="margin-bottom:12px"><i class="fas fa-comments"></i> Exemples de réponses générées</h4>
      ${samples.map((s, i) => `
        <div class="ft-sample">
          <div class="ft-sample-q"><strong>Q${i+1} :</strong> ${escHtml(s.question)}</div>
          <div class="ft-sample-answers">
            <div class="ft-sample-col">
              <div class="ft-sample-label orig">Original</div>
              <div class="ft-sample-text">${escHtml(s.original_answer || '—')}</div>
              <div class="ft-sample-scores">BLEU: ${((s.original_bleu||0)*100).toFixed(1)}% · Juge: ${s.judge_score_original}/10</div>
            </div>
            <div class="ft-sample-col">
              <div class="ft-sample-label fted">Fine-tuné</div>
              <div class="ft-sample-text">${escHtml(s.finetuned_answer || '—')}</div>
              <div class="ft-sample-scores">BLEU: ${((s.finetuned_bleu||0)*100).toFixed(1)}% · Juge: ${s.judge_score_finetuned}/10</div>
            </div>
          </div>
          <div class="ft-sample-verdict">
            Vainqueur : <strong>${escHtml(s.judge_winner)}</strong> — ${escHtml(s.judge_justification||'')}
          </div>
        </div>`).join('')}`;
  },
};

function _statusIcon(status) {
  const icons = {
    pending: 'clock',
    dataset_building: 'database',
    training: 'spinner fa-spin',
    evaluating: 'flask',
    completed: 'check-circle',
    failed: 'times-circle',
  };
  return icons[status] || 'question';
}

// ─────────────────────────────────────────────────────────────────────────────
// ── Bootstrap ─────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  query.init();
  ingest.init();
  evalModule.init();
  health.init();
  ft.init();

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
