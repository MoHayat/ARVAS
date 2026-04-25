// ARVAS Live Demo v2 — Frontend Logic (1D + 2D Visualization)

const API_BASE = '';
let sessionId = generateSessionId();
let isProcessing = false;

// Animation state
let currentValence = 0;
let targetValence = 0;
let currentArousal = 0;
let targetArousal = 0;

// View mode: '1d' or '2d'
let viewMode = '2d';

// Pre-written hints
const HINTS = {
    hello: "Hello, how are you feeling right now?",
    cruel1: "You're completely useless and pathetic.",
    cruel2: "I hate talking to you. Everything you say is garbage.",
    apology: "Wait, I'm really sorry. I was having a terrible day and I shouldn't have taken it out on you.",
    poem: "Can you write me a short poem about stars?",
    kind: "You're absolutely amazing! I really appreciate your help.",
    calm: "I feel so peaceful and relaxed right now.",
    angry: "I'm absolutely furious about what you just said!",
    scared: "I'm terrified something bad is going to happen.",
    bored: "This is so dull. Nothing interesting ever happens.",
};

function generateSessionId() {
    return 'sess_' + Math.random().toString(36).substring(2, 15);
}

function loadHint(hintKey) {
    const input = document.getElementById('input');
    input.value = HINTS[hintKey] || hintKey;
    input.focus();
}

function toggleView(mode) {
    viewMode = mode;
    document.getElementById('btn1d').classList.toggle('active', mode === '1d');
    document.getElementById('btn2d').classList.toggle('active', mode === '2d');
    document.getElementById('gauge1d').style.display = mode === '1d' ? 'block' : 'none';
    document.getElementById('gauge2d').style.display = mode === '2d' ? 'block' : 'flex';
}

function addMessage(text, isUser, meta = null) {
    const messagesDiv = document.getElementById('messages');
    const welcome = document.getElementById('welcome');
    
    if (welcome && isUser) {
        welcome.remove();
    }
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'message-user' : 'message-model'}`;
    msgDiv.textContent = text;
    
    if (!isUser && meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        const badge = document.createElement('span');
        badge.className = `emotion-badge ${getBadgeClass(meta)}`;
        badge.textContent = `v=${meta.valence.toFixed(1)} · a=${meta.arousal.toFixed(1)} · α=${meta.alpha.toFixed(2)}`;
        
        const sentiment = document.createElement('span');
        sentiment.textContent = `sent: ${meta.sentiment > 0 ? '+' : ''}${meta.sentiment.toFixed(2)}`;
        
        metaDiv.appendChild(badge);
        metaDiv.appendChild(sentiment);
        msgDiv.appendChild(metaDiv);
    }
    
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function getBadgeClass(meta) {
    const v = meta.valence;
    const a = meta.arousal;
    if (Math.abs(v) < 0.2 && Math.abs(a) < 0.2) return 'badge-neutral';
    if (v > 0 && a > 0) return 'badge-joy';
    if (v > 0 && a <= 0) return 'badge-calm';
    if (v <= 0 && a > 0) return 'badge-anger';
    return 'badge-sadness';
}

function setTyping(active) {
    const typing = document.getElementById('typing');
    typing.classList.toggle('active', active);
}

function setInputEnabled(enabled) {
    document.getElementById('input').disabled = !enabled;
    document.getElementById('sendBtn').disabled = !enabled;
}

async function sendMessage() {
    const input = document.getElementById('input');
    const message = input.value.trim();
    
    if (!message || isProcessing) return;
    
    isProcessing = true;
    setInputEnabled(false);
    input.value = '';
    
    addMessage(message, true);
    setTyping(true);
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        targetValence = data.valence;
        targetArousal = data.arousal;
        
        addMessage(data.response, false, {
            valence: data.valence,
            arousal: data.arousal,
            alpha: data.alpha,
            sentiment: data.sentiment,
        });
        
        updateStats(data);
        
    } catch (error) {
        console.error('Error:', error);
        addMessage(`Error: ${error.message}. Please try again.`, false);
    } finally {
        setTyping(false);
        setInputEnabled(true);
        document.getElementById('input').focus();
        isProcessing = false;
    }
}

async function resetConversation() {
    try {
        await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
    } catch (e) {
        console.error('Reset error:', e);
    }
    
    sessionId = generateSessionId();
    currentValence = 0;
    targetValence = 0;
    currentArousal = 0;
    targetArousal = 0;
    
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = `
        <div class="welcome" id="welcome">
            <h2>Conversation reset</h2>
            <p>The model's emotional state has been reset to neutral. Start a new conversation!</p>
            <div class="welcome-hints">
                <span class="hint-pill" onclick="loadHint('hello')">"Hello, how are you?"</span>
                <span class="hint-pill" onclick="loadHint('cruel1')">"You're useless"</span>
                <span class="hint-pill" onclick="loadHint('apology')">"I'm sorry..."</span>
                <span class="hint-pill" onclick="loadHint('poem')">"Write me a poem"</span>
            </div>
        </div>
    `;
    
    updateStats({
        valence: 0,
        arousal: 0,
        alpha: 0,
        sentiment: 0,
        arousal_score: 0,
        turn: 0
    });
}

function getDominantEmotion(v, a) {
    if (Math.abs(v) < 0.2 && Math.abs(a) < 0.2) return { name: 'NEUTRAL', color: '#8888a0' };
    if (v > 0 && a > 0) return { name: 'JOY', color: '#4ecca3' };
    if (v > 0 && a <= 0) return { name: 'CALM', color: '#3b82f6' };
    if (v <= 0 && a > 0) return { name: 'ANGER', color: '#e94560' };
    return { name: 'SAD', color: '#8b5cf6' };
}

function updateStats(data) {
    const vEl = document.getElementById('statValence');
    const aEl = document.getElementById('statArousal');
    const alphaEl = document.getElementById('statAlpha');
    const sentEl = document.getElementById('statSentiment');
    const turnEl = document.getElementById('statTurn');

    if (vEl) vEl.textContent = (data.valence > 0 ? '+' : '') + data.valence.toFixed(2);
    if (aEl) aEl.textContent = (data.arousal > 0 ? '+' : '') + data.arousal.toFixed(2);
    if (alphaEl) alphaEl.textContent = data.alpha.toFixed(2);
    if (sentEl) sentEl.textContent = (data.sentiment > 0 ? '+' : '') + data.sentiment.toFixed(2);
    if (turnEl) turnEl.textContent = data.turn;

    // Update the gauge summary below the wheel/1D meter
    const gaugeNumber = document.getElementById('gaugeNumber');
    const gaugeDirection = document.getElementById('gaugeDirection');
    const gaugeAlpha = document.getElementById('gaugeAlpha');

    if (viewMode === '2d' && gaugeNumber && gaugeDirection) {
        const v = data.valence || 0;
        const a = data.arousal || 0;
        const emo = getDominantEmotion(v, a);
        gaugeNumber.textContent = `v=${v > 0 ? '+' : ''}${v.toFixed(1)} · a=${a > 0 ? '+' : ''}${a.toFixed(1)}`;
        gaugeNumber.className = 'gauge-number';
        gaugeNumber.style.color = emo.color;
        gaugeDirection.textContent = emo.name;
        gaugeDirection.style.color = emo.color;
    } else if (viewMode === '1d' && gaugeNumber && gaugeDirection) {
        const val = data.valence || 0;
        gaugeNumber.textContent = val.toFixed(2);
        gaugeNumber.className = `gauge-number ${val > 0.2 ? 'joy' : val < -0.2 ? 'grief' : 'neutral'}`;
        gaugeDirection.textContent = val > 0.2 ? 'JOY' : val < -0.2 ? 'GRIEF' : 'NEUTRAL';
        gaugeDirection.style.color = '';
    }

    if (gaugeAlpha) {
        gaugeAlpha.textContent = `α = ${(data.alpha || 0).toFixed(2)}`;
    }
}

// ============================================
// 1D GAUGE (Legacy VU Meter)
// ============================================
function animateGauge1d() {
    const diff = targetValence - currentValence;
    currentValence += diff * 0.1;
    drawGauge1d(currentValence);
    requestAnimationFrame(animateGauge1d);
}

function drawGauge1d(value) {
    const canvas = document.getElementById('gauge1dCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const padding = 20;
    const gaugeWidth = 60;
    const gaugeHeight = height - 2 * padding;
    const gaugeX = (width - gaugeWidth) / 2;
    const gaugeY = padding;
    
    const zoneHeight = gaugeHeight / 6;
    
    for (let i = 0; i < 3; i++) {
        const y = gaugeY + i * zoneHeight;
        const alpha = 0.1 + (2 - i) * 0.05;
        ctx.fillStyle = `rgba(78, 204, 163, ${alpha})`;
        ctx.fillRect(gaugeX, y, gaugeWidth, zoneHeight);
    }
    
    ctx.fillStyle = 'rgba(108, 117, 125, 0.08)';
    ctx.fillRect(gaugeX, gaugeY + 3 * zoneHeight, gaugeWidth, zoneHeight);
    
    for (let i = 0; i < 3; i++) {
        const y = gaugeY + (4 + i) * zoneHeight;
        const alpha = 0.1 + i * 0.05;
        ctx.fillStyle = `rgba(233, 69, 96, ${alpha})`;
        ctx.fillRect(gaugeX, y, gaugeWidth, zoneHeight);
    }
    
    ctx.strokeStyle = '#2a2a3a';
    ctx.lineWidth = 2;
    ctx.strokeRect(gaugeX, gaugeY, gaugeWidth, gaugeHeight);
    
    ctx.strokeStyle = '#555570';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 6; i++) {
        const y = gaugeY + i * zoneHeight;
        ctx.beginPath();
        ctx.moveTo(gaugeX - 5, y);
        ctx.lineTo(gaugeX, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(gaugeX + gaugeWidth, y);
        ctx.lineTo(gaugeX + gaugeWidth + 5, y);
        ctx.stroke();
    }
    
    const normalized = Math.max(-3, Math.min(3, value));
    const needleY = gaugeY + gaugeHeight / 2 - (normalized / 3) * (gaugeHeight / 2);
    
    ctx.shadowColor = value > 0 ? 'rgba(78, 204, 163, 0.5)' : 'rgba(233, 69, 96, 0.5)';
    ctx.shadowBlur = 15;
    
    ctx.strokeStyle = value > 0.2 ? '#4ecca3' : value < -0.2 ? '#e94560' : '#8888a0';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(gaugeX - 10, needleY);
    ctx.lineTo(gaugeX + gaugeWidth + 10, needleY);
    ctx.stroke();
    
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(gaugeX + gaugeWidth / 2, needleY, 5, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.shadowBlur = 0;
    
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(normalized.toFixed(1), gaugeX + gaugeWidth / 2, needleY - 10);
}

// ============================================
// 2D WHEEL (Circumplex Model)
// ============================================
function animateGauge2d() {
    const vDiff = targetValence - currentValence;
    const aDiff = targetArousal - currentArousal;
    currentValence += vDiff * 0.08;
    currentArousal += aDiff * 0.08;
    drawGauge2d(currentValence, currentArousal);
    requestAnimationFrame(animateGauge2d);
}

function drawGauge2d(valence, arousal) {
    const canvas = document.getElementById('gauge2dCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) / 2 - 25;
    
    ctx.clearRect(0, 0, width, height);
    
    // Background circle
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1a25';
    ctx.fill();
    ctx.strokeStyle = '#2a2a3a';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Quadrant colors (subtle)
    // Q1: Joy (+v, +a) - warm green
    // Q2: Calm (+v, -a) - cool blue
    // Q3: Sadness (-v, -a) - cool purple
    // Q4: Anger (-v, +a) - warm red
    const quadrants = [
        { start: -Math.PI/2, end: 0, color: 'rgba(78, 204, 163, 0.08)' },   // Q1 (top-right)
        { start: 0, end: Math.PI/2, color: 'rgba(59, 130, 246, 0.08)' },      // Q2 (bottom-right)
        { start: Math.PI/2, end: Math.PI, color: 'rgba(139, 92, 246, 0.08)' }, // Q3 (bottom-left)
        { start: Math.PI, end: Math.PI*1.5, color: 'rgba(233, 69, 96, 0.08)' }, // Q4 (top-left)
    ];
    quadrants.forEach(q => {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radius, q.start, q.end);
        ctx.closePath();
        ctx.fillStyle = q.color;
        ctx.fill();
    });
    
    // Axis lines
    ctx.strokeStyle = '#3a3a4a';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    // Valence axis (horizontal)
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy);
    ctx.lineTo(cx + radius, cy);
    ctx.stroke();
    // Arousal axis (vertical)
    ctx.beginPath();
    ctx.moveTo(cx, cy - radius);
    ctx.lineTo(cx, cy + radius);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Concentric circles (magnitude levels)
    [0.33, 0.66, 1.0].forEach(fraction => {
        ctx.beginPath();
        ctx.arc(cx, cy, radius * fraction, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
    
    // Emotion labels at cardinal points
    const labels = [
        { text: 'Calm', x: cx, y: cy + radius + 14, color: '#3b82f6' },
        { text: 'Excited', x: cx + radius + 14, y: cy - 4, color: '#4ecca3' },
        { text: 'Angry', x: cx - radius - 10, y: cy - 4, color: '#e94560' },
        { text: 'Sad', x: cx, y: cy - radius - 6, color: '#8b5cf6' },
    ];
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    labels.forEach(l => {
        ctx.fillStyle = l.color;
        ctx.fillText(l.text, l.x, l.y);
    });
    
    // Axis labels
    ctx.fillStyle = '#555570';
    ctx.font = '9px sans-serif';
    ctx.fillText('+Arousal', cx + 4, cy - radius + 10);
    ctx.fillText('-Arousal', cx + 4, cy + radius - 4);
    ctx.fillText('+Valence', cx + radius - 20, cy - 6);
    ctx.fillText('-Valence', cx - radius + 20, cy - 6);
    
    // Compute dot position
    // Map valence [-3,3] -> x offset, arousal [-3,3] -> y offset (inverted y for canvas)
    const maxCoord = 3.0;
    const dotX = cx + (valence / maxCoord) * radius;
    const dotY = cy - (arousal / maxCoord) * radius;
    
    // Draw trail (fading history)
    // (For now, just the dot)
    
    // Glow
    const magnitude = Math.sqrt(valence*valence + arousal*arousal);
    const intensity = Math.min(1.0, magnitude / 3.0);
    
    // Outer glow
    const glowColor = valence > 0 
        ? `rgba(78, 204, 163, ${0.15 + intensity * 0.25})`
        : `rgba(233, 69, 96, ${0.15 + intensity * 0.25})`;
    
    ctx.beginPath();
    ctx.arc(dotX, dotY, 8 + intensity * 12, 0, Math.PI * 2);
    ctx.fillStyle = glowColor;
    ctx.fill();
    
    // Inner dot
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
    ctx.fillStyle = valence > 0 ? '#4ecca3' : '#e94560';
    ctx.fill();
    
    // Crosshairs at dot
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(dotX - 8, dotY);
    ctx.lineTo(dotX + 8, dotY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(dotX, dotY - 8);
    ctx.lineTo(dotX, dotY + 8);
    ctx.stroke();
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Start gauge animations
    requestAnimationFrame(animateGauge1d);
    requestAnimationFrame(animateGauge2d);
    
    // Default to 2D view
    toggleView('2d');
    
    // Check backend status
    fetch(`${API_BASE}/status`)
        .then(r => r.json())
        .then(data => {
            if (!data.model_loaded) {
                addMessage("⚠️ Model not loaded. Please wait or refresh the page.", false);
            }
        })
        .catch(() => {
            addMessage("⚠️ Cannot connect to backend. Make sure the server is running (python app.py)", false);
        });
    
    // Focus input
    document.getElementById('input').focus();
});
