import { BACKEND_URL, POLL_INTERVAL_MS, MODEL_NAME } from "./config.js";

// ─── Session ──────────────────────────────────────────────────────────────────
function getOrCreateSessionId() {
    let id = localStorage.getItem("talk_session_id");
    if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem("talk_session_id", id);
    }
    return id;
}

const SESSION_ID = getOrCreateSessionId();

// ─── DOM refs ────────────────────────────────────────────────────────────────
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const chatLog = document.getElementById("chat-log");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const metricsEl = document.getElementById("metrics-bar");

// Logs DOM
const logsToggle = document.getElementById("Logs-toggle") || document.getElementById("logs-toggle");
const logsPanel = document.getElementById("logs-panel");
const logsContent = document.getElementById("logs-content");
const logsEmpty = document.getElementById("logs-empty-state");

// ─── Status ──────────────────────────────────────────────────────────────────
function setStatus(state) {
    if (state === "online") {
        statusDot.className = "status-dot online";
        statusLabel.textContent = "online";
        document.body.classList.remove("offline");
    } else {
        statusDot.className = "status-dot offline";
        statusLabel.textContent = "offline";
        document.body.classList.add("offline");
    }
}

async function checkHealth() {
    try {
        const res = await fetch(`${BACKEND_URL}/health`, {
            signal: AbortSignal.timeout(5000),
        });
        setStatus(res.ok ? "online" : "offline");
    } catch {
        setStatus("offline");
    }
}

checkHealth();
setInterval(checkHealth, POLL_INTERVAL_MS);

// ─── Render messages ─────────────────────────────────────────────────────────
function renderMessage(role, text) {
    const wrap = document.createElement("div");
    wrap.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "user") {
        bubble.textContent = text;
    } else {
        bubble.innerHTML = window.marked ? window.marked.parse(text) : text;
    }

    wrap.appendChild(bubble);
    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function renderThinking() {
    const wrap = document.createElement("div");
    wrap.className = "message assistant thinking-wrap";
    wrap.id = "thinking-indicator";

    const bubble = document.createElement("div");
    bubble.className = "bubble thinking";
    bubble.innerHTML = `
    <span class="dot"></span>
    <span class="dot"></span>
    <span class="dot"></span>`;

    wrap.appendChild(bubble);
    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
    return wrap;
}

function updateMetrics(metrics) {
    if (!metrics || !metricsEl) return;
    metricsEl.style.display = "";
    metricsEl.textContent = `${metrics.latency_ms} ms · ${metrics.output_tokens} tokens · ${metrics.tokens_per_second} t/s`;
    metricsEl.style.opacity = "1";
}

// ─── Send ─────────────────────────────────────────────────────────────────────
async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = "";
    inputEl.style.height = "auto";
    sendBtn.disabled = true;
    metricsEl.style.opacity = "0";

    renderMessage("user", text);
    const thinkEl = renderThinking();

    try {
        const res = await fetch(`${BACKEND_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text, session_id: SESSION_ID }),
            // Give enough timeout for streaming responses
            signal: AbortSignal.timeout(120_000),
        });

        thinkEl.remove();

        if (!res.ok) {
            renderMessage("assistant", `⚠ Server error (${res.status}). Try again.`);
            return;
        }

        // Create an empty assistant message block to append tokens to
        const wrap = document.createElement("div");
        wrap.className = "message assistant";
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        wrap.appendChild(bubble);
        chatLog.appendChild(wrap);

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;
        let partialText = "";
        let fullMarkdown = "";

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;

            if (value) {
                partialText += decoder.decode(value, { stream: true });
                const lines = partialText.split("\n");

                // Keep the last incomplete line in the buffer
                partialText = lines.pop();

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const payload = JSON.parse(line.slice(6));

                        if (payload.type === "token") {
                            fullMarkdown += payload.content;

                            // Re-render the chat bubble markdown progressively
                            bubble.innerHTML = window.marked ? window.marked.parse(fullMarkdown) : fullMarkdown;
                            chatLog.scrollTop = chatLog.scrollHeight;
                        } else if (payload.type === "metrics") {
                            updateMetrics(payload.content);
                        }
                    }
                }
            }
        }
    } catch (err) {
        thinkEl.remove();
        renderMessage(
            "assistant",
            "⚠ Could not reach the backend. The laptop may be offline."
        );
    } finally {
        sendBtn.disabled = false;
        inputEl.focus();
    }
}

// ─── Event listeners ─────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendMessage);

inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
function adjustInputHeight() {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
}
inputEl.addEventListener("input", adjustInputHeight);

let lastChatAreaWidth = 0;
const resizeObserver = new ResizeObserver((entries) => {
    for (let entry of entries) {
        if (entry.contentRect.width !== lastChatAreaWidth) {
            lastChatAreaWidth = entry.contentRect.width;
            if (inputEl.value) {
                adjustInputHeight();
            }
        }
    }
});
resizeObserver.observe(document.getElementById("chat-area"));

// ─── Sidebar toggle (mobile) ──────────────────────────────────────────────────
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebar = document.getElementById("sidebar");

sidebarToggle?.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    logsPanel.classList.remove("open");
});

// ─── Logs Panel ───────────────────────────────────────────────────────────────
logsToggle?.addEventListener("click", () => {
    if (window.innerWidth <= 768) {
        logsPanel.classList.toggle("open");
        sidebar.classList.remove("open");
    } else {
        logsPanel.classList.toggle("closed");
    }
});

// Setup Live Logs stream
const logsSource = new EventSource(`${BACKEND_URL}/logs/stream`);
logsSource.onmessage = (event) => {
    if (logsEmpty) {
        logsEmpty.remove();
    }
    const logText = event.data;
    if (!logText) return;

    let cleanText = logText.trim();
    let isMetricsLog = false;
    let entry = document.createElement("div");

    if (cleanText.startsWith("{") && cleanText.endsWith("}")) {
        // Convert Python dictionary format to valid JSON string
        let jsonStr = cleanText
            .replace(/'/g, '"')
            .replace(/:\s*True/g, ': true')
            .replace(/:\s*False/g, ': false')
            .replace(/:\s*None/g, ': null');
        try {
            const data = JSON.parse(jsonStr);
            isMetricsLog = true;

            const date = new Date(data.timestamp);
            const formattedTime = isNaN(date.getTime())
                ? data.timestamp
                : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });

            const gpuUtil = data.gpu_util_percent === -1 ? 'N/A' : `${data.gpu_util_percent}%`;
            const latency = data.latency_ms ? `${data.latency_ms.toFixed(0)} ms` : '—';
            
            entry.className = "log-card metrics-card";
            entry.innerHTML = `
                <div class="card-header">
                    <span class="card-tag"><span class="pulse-indicator"></span>Inference Metrics</span>
                    <span class="card-time">${formattedTime}</span>
                </div>
                <div class="card-metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Latency</span>
                        <span class="metric-value highlight">${latency}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Prompt Tokens</span>
                        <span class="metric-value">${data.prompt_tokens ?? '—'}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">RAM Used</span>
                        <span class="metric-value">${data.ram_used_gb ? data.ram_used_gb + ' GB' : '—'}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">VRAM Peak</span>
                        <span class="metric-value">${data.vram_used_mb ? data.vram_used_mb + ' MB' : '—'}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">VRAM Delta</span>
                        <span class="metric-value">${data.vram_delta_mb ? data.vram_delta_mb + ' MB' : '—'}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">GPU Util</span>
                        <span class="metric-value">${gpuUtil}</span>
                    </div>
                </div>
            `;
        } catch (e) {
            isMetricsLog = false;
        }
    }

    if (!isMetricsLog) {
        entry.className = "log-entry";
        if (logText.toLowerCase().includes("error") || logText.toLowerCase().includes("exception")) {
            entry.classList.add("error");
        }
        entry.textContent = logText;
    }

    logsContent.appendChild(entry);

    // Auto-scroll to bottom if near bottom
    if (logsContent.scrollHeight - logsContent.scrollTop - logsContent.clientHeight < 200) {
        logsContent.scrollTop = logsContent.scrollHeight;
    }
};

logsSource.onerror = () => {
    // silently reconnects by default
};
