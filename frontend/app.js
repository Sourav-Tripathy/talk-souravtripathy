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

    // Light markdown: wrap code blocks in <pre><code>
    const escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    bubble.innerHTML = escaped.replace(
        /```([\s\S]*?)```/g,
        (_, code) => `<pre><code>${code.trim()}</code></pre>`
    );

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
    if (!metrics) return;
    metricsEl.textContent =
        `${metrics.latency_ms} ms · ${metrics.vram_used_mb} MB VRAM · ~${metrics.prompt_tokens} tokens`;
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
                            const escaped = fullMarkdown
                                .replace(/&/g, "&amp;")
                                .replace(/</g, "&lt;")
                                .replace(/>/g, "&gt;");
                                
                            bubble.innerHTML = escaped.replace(
                                /```([\s\S]*?)```/g,
                                (_, code) => `<pre><code>${code.trim()}</code></pre>`
                            );
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
inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
});

// ─── Sidebar toggle (mobile) ──────────────────────────────────────────────────
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebar = document.getElementById("sidebar");

sidebarToggle?.addEventListener("click", () => {
    sidebar.classList.toggle("open");
});
