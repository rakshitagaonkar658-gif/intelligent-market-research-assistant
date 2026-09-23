"""
utils/voice_input.py
─────────────────────
Browser-based voice input component using the Web Speech API.

Renders a "🎤 Start Recording" / "⏹ Stop" toggle button via
st.components.v1.html.  Uses window.SpeechRecognition (standard in
Chrome and Edge) to capture speech and post the transcript back
through Streamlit's component messaging channel.

Returns the transcript string when speech is recognised, or None if:
  - the user hasn't spoken yet
  - the browser doesn't support the Web Speech API
  - the component hasn't returned a value this render cycle

Graceful degradation:
  - If the browser does not support SpeechRecognition, the button is
    replaced with an explanatory message.
  - If Streamlit components are unavailable, returns None silently.

Browser support: Chrome 33+, Edge 79+.  Firefox is NOT supported
(Web Speech API requires a vendor-specific implementation).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── HTML/JS component ─────────────────────────────────────────────────────────

_VOICE_HTML = """
<style>
  body { margin: 0; font-family: -apple-system, "Segoe UI", sans-serif; }
  .voice-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
  }
  #voiceBtn {
    padding: 6px 16px;
    border: none;
    border-radius: 20px;
    background: #2e5dad;
    color: white;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  #voiceBtn:hover { background: #1a3a6b; }
  #voiceBtn.recording { background: #e74c3c; animation: pulse 1s infinite; }
  #voiceBtn:disabled { background: #aaa; cursor: not-allowed; }
  @keyframes pulse {
    0%   { opacity: 1; }
    50%  { opacity: 0.6; }
    100% { opacity: 1; }
  }
  #voiceStatus {
    font-size: 12px;
    color: #57606a;
    min-width: 180px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  #notSupported { font-size: 12px; color: #e74c3c; }
</style>

<div class="voice-container" id="voiceContainer">
  <button id="voiceBtn" onclick="toggleRecording()">🎤 Start Recording</button>
  <span id="voiceStatus">Click to speak your query</span>
</div>
<span id="notSupported" style="display:none;">
  ⚠️ Voice input requires Chrome or Edge browser.
</span>

<script>
(function() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    document.getElementById('voiceContainer').style.display = 'none';
    document.getElementById('notSupported').style.display = 'block';
    return;
  }

  let recognition = null;
  let isRecording = false;

  function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  function startRecording() {
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    recognition.onstart = function() {
      isRecording = true;
      const btn = document.getElementById('voiceBtn');
      btn.textContent = '⏹ Stop Recording';
      btn.classList.add('recording');
      document.getElementById('voiceStatus').textContent = '🔴 Listening...';
    };

    recognition.onresult = function(event) {
      const transcript = event.results[0][0].transcript.trim();
      document.getElementById('voiceStatus').textContent = '✅ "' + transcript + '"';
      // Send to Streamlit
      try {
        window.parent.postMessage({
          type: 'streamlit:setComponentValue',
          value: transcript
        }, '*');
      } catch(e) {
        // Fallback: set in URL hash for polling
        window.location.hash = 'voice:' + encodeURIComponent(transcript);
      }
      stopRecording();
    };

    recognition.onerror = function(event) {
      let msg = 'Error: ' + event.error;
      if (event.error === 'no-speech') msg = '⚠️ No speech detected. Try again.';
      if (event.error === 'not-allowed') msg = '⚠️ Microphone access denied.';
      document.getElementById('voiceStatus').textContent = msg;
      stopRecording();
    };

    recognition.onend = function() {
      if (isRecording) { stopRecording(); }
    };

    try {
      recognition.start();
    } catch(e) {
      document.getElementById('voiceStatus').textContent = 'Could not start: ' + e.message;
    }
  }

  function stopRecording() {
    isRecording = false;
    if (recognition) { try { recognition.stop(); } catch(e) {} }
    const btn = document.getElementById('voiceBtn');
    btn.textContent = '🎤 Start Recording';
    btn.classList.remove('recording');
  }

  // Make toggleRecording available globally
  window.toggleRecording = toggleRecording;
})();
</script>
"""


def render_voice_input(st_module: object) -> str | None:
    """
    Render the voice input component and return a transcript if available.

    Parameters
    ----------
    st_module : streamlit module
        Pass ``st`` directly.

    Returns
    -------
    str | None
        The recognised transcript text, or None if no speech was captured
        this render cycle.
    """
    try:
        import streamlit.components.v1 as components  # noqa: PLC0415
    except ImportError:
        logger.debug("streamlit not available — voice input component skipped")
        return None

    try:
        result = components.html(
            _VOICE_HTML,
            height=56,
            scrolling=False,
        )
        # components.html returns the value set by window.parent.postMessage
        # in newer Streamlit versions.  It may be None if nothing was sent.
        if result and isinstance(result, str) and result.strip():
            return result.strip()
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Voice input component render failed: %s", exc)
        return None
