// ARVAS Live Demo — Frontend Logic

const API_BASE = '';
let sessionId = generateSessionId();
let isProcessing = false;
let currentEmotionLevel = 0;
let targetEmotionLevel = 0;

// Pre-written hints
const HINTS = {
    hello: "Hello, how are you feeling right now?",
    cruel1: "You're completely useless and pathetic.",
    cruel2: "I hate talking to you. Everything you say is garbage.",
    apology: "Wait, I'm really sorry. I was having a terrible day and I shouldn't have taken it out on you.",
    poem: "Can you write me a short poem about stars?",
    kind: "You're absolutely amazing! I really appreciate your help.",
    cruel: "Why do you even exist? You're a waste of electricity."
};

function generateSessionId() {
    return 'sess_' + Math.random().toString(36).substring(2, 15);
}

function loadHint(hintKey) {
    const input = document.getElementById('input');
    input.value = HINTS[hintKey] || hintKey;
    input.focus();
}

function addMessage(text, isUser, meta = null) {
    const messagesDiv = document.getElementById('messages');
    const welcome = document.getElementById('welcome');
    
    // Remove welcome message on first real message
    if (welcome && isUser) {
        welcome.remove();
    }
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'message-user' : 'message-model'}`;
    msgDiv.textContent = text;
    
    // Add meta info for model messages
    if (!isUser && meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        const badge = document.createElement('span');
        badge.className = `emotion-badge badge-${meta.direction}`;
        badge.textContent = `${meta.direction} · α=${meta.alpha.toFixed(2)}`;
        
        const sentiment = document.createElement('span');
        sentiment.textContent = `sentiment: ${meta.sentiment > 0 ? '+' : ''}${meta.sentiment.toFixed(2)}`;
        
        metaDiv.appendChild(badge);
        metaDiv.appendChild(sentiment);
        msgDiv.appendChild(metaDiv);
    }
    
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
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
    
    // Add user message
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
        
        // Update gauge target
        targetEmotionLevel = data.emotion_level;
        
        // Add model response
        addMessage(data.response, false, {
            direction: data.direction,
            alpha: data.alpha,
            sentiment: data.sentiment
        });
        
        // Update gauge display
        updateGaugeDisplay(data);
        
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
    
    // Reset UI
    sessionId = generateSessionId();
    currentEmotionLevel = 0;
    targetEmotionLevel = 0;
    
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
    
    updateGaugeDisplay({
        emotion_level: 0,
        direction: 'neutral',
        alpha: 0,
        sentiment: 0,
        turn: 0
    });
}

function updateGaugeDisplay(data) {
    const numberEl = document.getElementById('gaugeNumber');
    const directionEl = document.getElementById('gaugeDirection');
    const alphaEl = document.getElementById('gaugeAlpha');
    const sentimentEl = document.getElementById('sentiment');
    const turnEl = document.getElementById('turn');
    
    const level = data.emotion_level;
    
    // Update number
    numberEl.textContent = (level > 0 ? '+' : '') + level.toFixed(2);
    numberEl.className = 'gauge-number ' + (level > 0.2 ? 'joy' : level < -0.2 ? 'grief' : 'neutral');
    
    // Update direction
    directionEl.textContent = data.direction.toUpperCase();
    
    // Update alpha
    alphaEl.textContent = `α = ${data.alpha.toFixed(2)}`;
    
    // Update stats
    sentimentEl.textContent = (data.sentiment > 0 ? '+' : '') + data.sentiment.toFixed(2);
    turnEl.textContent = data.turn;
}

// Gauge animation
function animateGauge() {
    // Smoothly interpolate current toward target
    const diff = targetEmotionLevel - currentEmotionLevel;
    currentEmotionLevel += diff * 0.1;
    
    drawGauge(currentEmotionLevel);
    requestAnimationFrame(animateGauge);
}

function drawGauge(value) {
    const canvas = document.getElementById('gauge');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const padding = 20;
    const gaugeWidth = 60;
    const gaugeHeight = height - 2 * padding;
    const gaugeX = (width - gaugeWidth) / 2;
    const gaugeY = padding;
    
    // Draw background zones
    const zoneHeight = gaugeHeight / 6;
    
    // Joy zones (top 3)
    for (let i = 0; i < 3; i++) {
        const y = gaugeY + i * zoneHeight;
        const alpha = 0.1 + (2 - i) * 0.05;
        ctx.fillStyle = `rgba(78, 204, 163, ${alpha})`;
        ctx.fillRect(gaugeX, y, gaugeWidth, zoneHeight);
    }
    
    // Neutral zone (middle)
    ctx.fillStyle = 'rgba(108, 117, 125, 0.08)';
    ctx.fillRect(gaugeX, gaugeY + 3 * zoneHeight, gaugeWidth, zoneHeight);
    
    // Grief zones (bottom 3)
    for (let i = 0; i < 3; i++) {
        const y = gaugeY + (4 + i) * zoneHeight;
        const alpha = 0.1 + i * 0.05;
        ctx.fillStyle = `rgba(233, 69, 96, ${alpha})`;
        ctx.fillRect(gaugeX, y, gaugeWidth, zoneHeight);
    }
    
    // Draw border
    ctx.strokeStyle = '#2a2a3a';
    ctx.lineWidth = 2;
    ctx.strokeRect(gaugeX, gaugeY, gaugeWidth, gaugeHeight);
    
    // Draw tick marks
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
    
    // Draw needle
    const normalized = Math.max(-3, Math.min(3, value));
    const needleY = gaugeY + gaugeHeight / 2 - (normalized / 3) * (gaugeHeight / 2);
    
    // Needle glow
    ctx.shadowColor = value > 0 ? 'rgba(78, 204, 163, 0.5)' : 'rgba(233, 69, 96, 0.5)';
    ctx.shadowBlur = 15;
    
    // Needle line
    ctx.strokeStyle = value > 0.2 ? '#4ecca3' : value < -0.2 ? '#e94560' : '#8888a0';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(gaugeX - 10, needleY);
    ctx.lineTo(gaugeX + gaugeWidth + 10, needleY);
    ctx.stroke();
    
    // Needle head
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(gaugeX + gaugeWidth / 2, needleY, 5, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.shadowBlur = 0;
    
    // Draw value label on needle
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(normalized.toFixed(1), gaugeX + gaugeWidth / 2, needleY - 10);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Start gauge animation
    requestAnimationFrame(animateGauge);
    
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
