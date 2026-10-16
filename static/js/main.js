/**
 * main.js
 * =======
 * HarmoniQ AI - Frontend Controller & WebAudio MIDI Synthesizer
 * Connects UI to Flask API (/api/health, /api/generate, /api/download)
 * and renders synthesized in-browser audio via Tone.js & @tonejs/midi.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------------------
    // Health / Header
    const backendStatusPill = document.getElementById("backendStatusPill");
    const backendStatusText = document.getElementById("backendStatusText");
    const infoVocab = document.getElementById("infoVocab");
    const infoSeq = document.getElementById("infoSeq");

    // Form Controls
    const generateForm = document.getElementById("generateForm");
    const generateBtn = document.getElementById("generateBtn");
    const numNotesInput = document.getElementById("numNotesInput");
    const numNotesBadge = document.getElementById("numNotesBadge");
    const tempInput = document.getElementById("tempInput");
    const tempBadge = document.getElementById("tempBadge");
    const presetChips = document.querySelectorAll(".chip");

    // Status / Alerts
    const statusBanner = document.getElementById("statusBanner");
    const errorAlert = document.getElementById("errorAlert");
    const errorMessage = document.getElementById("errorMessage");
    const dismissErrorBtn = document.getElementById("dismissErrorBtn");

    // Audio Player & Controls
    const playerBadge = document.getElementById("playerBadge");
    const visualizerCanvas = document.getElementById("visualizerCanvas");
    const visualizerOverlay = document.getElementById("visualizerOverlay");
    const overlayMessage = document.getElementById("overlayMessage");
    const playBtn = document.getElementById("playBtn");
    const playIcon = document.getElementById("playIcon");
    const pauseIcon = document.getElementById("pauseIcon");
    const stopBtn = document.getElementById("stopBtn");
    const seekSlider = document.getElementById("seekSlider");
    const currentTimeLabel = document.getElementById("currentTimeLabel");
    const totalTimeLabel = document.getElementById("totalTimeLabel");
    const volumeSlider = document.getElementById("volumeSlider");
    const downloadBtn = document.getElementById("downloadBtn");
    const downloadBtnText = document.getElementById("downloadBtnText");

    // Metadata Display
    const metadataCard = document.getElementById("metadataCard");
    const metaFilename = document.getElementById("metaFilename");
    const metaDuration = document.getElementById("metaDuration");
    const metaNotes = document.getElementById("metaNotes");
    const metaGenTime = document.getElementById("metaGenTime");

    // -------------------------------------------------------------------------
    // Audio Player State & WebAudio Synthesizer
    // -------------------------------------------------------------------------
    let currentMidiData = null;
    let scheduledNotes = [];
    let isPlaying = false;
    let playbackStartTime = 0;
    let pauseOffset = 0;
    let playbackDuration = 0;
    let animationFrameId = null;
    let toneSynth = null;
    let activeVisualizerNotes = [];

    // Initialize Tone.js Synthesizer
    function initSynthesizer() {
        if (typeof Tone === "undefined") {
            console.warn("[Player] Tone.js not detected. In-browser synthesis unavailable.");
            return false;
        }

        if (!toneSynth) {
            try {
                // Polyphonic Synth with rich piano/electric organ envelope
                toneSynth = new Tone.PolySynth(Tone.Synth, {
                    oscillator: { type: "triangle8" },
                    envelope: {
                        attack: 0.005,
                        decay: 0.18,
                        sustain: 0.25,
                        release: 0.75
                    }
                });

                // Gentle reverb for spatial warmth
                const reverb = new Tone.Reverb({ decay: 1.5, preDelay: 0.01, wet: 0.2 }).toDestination();
                toneSynth.connect(reverb);

                console.log("[Player] WebAudio polyphonic synth initialized.");
            } catch (err) {
                console.error("[Player] Failed to create Tone.js synth:", err);
                return false;
            }
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Safe API Response Parser & Error Extraction Helpers
    // -------------------------------------------------------------------------
    /**
     * Safely parses any HTTP response without throwing, correctly distinguishing
     * valid JSON objects, HTML error pages, plain-text messages, and empty responses.
     */
    async function parseApiResponse(response) {
        let rawText = "";
        try {
            rawText = await response.text();
        } catch (readErr) {
            return { isJson: false, data: null, rawText: "" };
        }

        if (!rawText || !rawText.trim()) {
            return { isJson: false, data: null, rawText: "" };
        }

        const trimmed = rawText.trim();
        const contentType = (response.headers && response.headers.get("content-type")) || "";

        if (contentType.includes("application/json") || trimmed.startsWith("{") || trimmed.startsWith("[")) {
            try {
                const data = JSON.parse(trimmed);
                return { isJson: true, data: data, rawText: trimmed };
            } catch (jsonErr) {
                return { isJson: false, data: null, rawText: trimmed };
            }
        }

        return { isJson: false, data: null, rawText: trimmed };
    }

    /**
     * Extracts a clear, user-friendly error description from any response format.
     */
    function extractErrorMessage(response, parseResult, defaultMsg) {
        // 1. Valid JSON with a descriptive error/message field
        if (parseResult && parseResult.isJson && parseResult.data) {
            if (typeof parseResult.data.message === "string" && parseResult.data.message.trim()) {
                return parseResult.data.message.trim();
            }
            if (typeof parseResult.data.error === "string" && parseResult.data.error.trim()) {
                return parseResult.data.error.trim();
            }
        }

        // 2. HTML Error pages (e.g., from proxies, servers, or unhandled 500s)
        if (parseResult && parseResult.rawText) {
            const text = parseResult.rawText.trim();
            if (text.startsWith("<") && text.includes(">")) {
                const titleMatch = text.match(/<title>(.*?)<\/title>/i);
                if (titleMatch && titleMatch[1]) {
                    const cleanTitle = titleMatch[1].replace(/<[^>]*>?/gm, "").trim();
                    return `Server Error (${response ? response.status : '500'}): ${cleanTitle}`;
                }
                return `Server Error (${response ? response.status : '500'}). Please check backend logs.`;
            }
            // Short plain-text error message
            if (text.length > 0 && text.length <= 250) {
                return text;
            }
        }

        // 3. Fallback based on HTTP status code
        if (response) {
            if (response.status === 400) return "Bad request. Please verify your parameter inputs.";
            if (response.status === 404) return "Requested API endpoint was not found on the server.";
            if (response.status === 500) return "The AI music engine encountered an internal server error.";
            if (response.status === 502 || response.status === 504) return "Server gateway error. The backend may be restarting.";
            if (response.status === 503) return "The AI music service is temporarily unavailable.";
            if (!response.ok) return `Server error (${response.status} ${response.statusText || 'Error'}).`;
        }

        return defaultMsg || "An unexpected error occurred.";
    }

    /**
     * Translates low-level browser network failure exceptions into friendly guidance.
     */
    function formatNetworkError(err) {
        if (!err) return "Network connection failed. Please ensure the Flask backend is running.";
        const msg = (err.message || "").toLowerCase();
        if (
            msg.includes("failed to fetch") ||
            msg.includes("networkerror") ||
            msg.includes("connection refused") ||
            msg.includes("load failed") ||
            msg.includes("err_connection")
        ) {
            return "Cannot connect to the server. Please verify the Flask backend is running on http://127.0.0.1:5000.";
        }
        return err.message || "Network request failed. Please check your connection.";
    }

    // -------------------------------------------------------------------------
    // 1. Backend Health Check
    // -------------------------------------------------------------------------
    async function checkBackendHealth() {
        backendStatusPill.className = "status-pill status-checking";
        backendStatusText.textContent = "Connecting...";

        try {
            let res;
            try {
                res = await fetch("/api/health");
            } catch (networkErr) {
                throw new Error(formatNetworkError(networkErr));
            }

            const parsed = await parseApiResponse(res);

            if (!res.ok) {
                const errMsg = extractErrorMessage(res, parsed, `Server returned HTTP ${res.status}`);
                throw new Error(errMsg);
            }

            if (!parsed.isJson || !parsed.data) {
                throw new Error("Server returned an invalid or empty response instead of JSON.");
            }

            const data = parsed.data;
            if (data.status === "healthy") {
                backendStatusPill.className = "status-pill status-online";
                backendStatusText.textContent = "Backend Online • Ready";
                backendStatusPill.title = `Model: LSTM (Context: ${data.sequence_length || 20} notes, Vocabulary: ${data.unique_pitches_count || 71} pitches)`;

                if (infoVocab && data.unique_pitches_count) {
                    infoVocab.textContent = `${data.unique_pitches_count} Pitches`;
                }
                if (infoSeq && data.sequence_length) {
                    infoSeq.textContent = `${data.sequence_length} Notes`;
                }
            } else {
                throw new Error(data.message || "Server reported unhealthy status");
            }
        } catch (err) {
            console.warn("[Health] Failed to reach backend:", err);
            backendStatusPill.className = "status-pill status-offline";
            backendStatusText.textContent = "Backend Offline (Retry)";
            backendStatusPill.title = err.message || "Click to retry connecting to the Flask server";
        }
    }

    backendStatusPill.addEventListener("click", checkBackendHealth);
    checkBackendHealth();

    // -------------------------------------------------------------------------
    // 2. Controls & Sliders Synchronization
    // -------------------------------------------------------------------------
    function updateBadges() {
        numNotesBadge.textContent = `${numNotesInput.value} notes`;
        tempBadge.textContent = Number(tempInput.value).toFixed(2);
    }

    numNotesInput.addEventListener("input", () => {
        updateBadges();
        syncChipState(numNotesInput);
    });

    tempInput.addEventListener("input", () => {
        updateBadges();
        syncChipState(tempInput);
    });

    // Preset Chips
    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const targetId = chip.dataset.target;
            const targetVal = chip.dataset.val;
            const targetInput = document.getElementById(targetId);

            if (targetInput) {
                targetInput.value = targetVal;
                updateBadges();
                syncChipState(targetInput);
            }
        });
    });

    function syncChipState(inputElement) {
        presetChips.forEach(chip => {
            if (chip.dataset.target === inputElement.id) {
                if (parseFloat(chip.dataset.val) === parseFloat(inputElement.value)) {
                    chip.classList.add("active");
                } else {
                    chip.classList.remove("active");
                }
            }
        });
    }

    // Dismiss Error Banner
    dismissErrorBtn.addEventListener("click", () => {
        errorAlert.classList.add("hidden");
    });

    // -------------------------------------------------------------------------
    // 3. Generate Music Action (POST /api/generate)
    // -------------------------------------------------------------------------
    generateForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await handleGenerate();
    });

    async function handleGenerate() {
        // Clear previous errors and stop playback
        stopPlayback();
        errorAlert.classList.add("hidden");

        const numNotes = parseInt(numNotesInput.value, 10);
        const temperature = parseFloat(tempInput.value);

        // UI Loading State: disable button and display active progress banner
        generateBtn.disabled = true;
        statusBanner.classList.remove("hidden");
        playerBadge.textContent = "Generating...";
        playerBadge.style.color = "#a78bfa";

        try {
            console.log(`[Generate] Sending request: num_notes=${numNotes}, temperature=${temperature}`);

            let response;
            try {
                response = await fetch("/api/generate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        num_notes: numNotes,
                        temperature: temperature
                    })
                });
            } catch (networkErr) {
                // Catches network dropouts, offline state, connection refused
                throw new Error(formatNetworkError(networkErr));
            }

            // Safely parse response (handles JSON, HTML errors, plain text, and empty bodies)
            const parsed = await parseApiResponse(response);

            // Check HTTP status code before treating as success
            if (!response.ok) {
                const errorMsg = extractErrorMessage(response, parsed, `Generation failed (HTTP ${response.status})`);
                throw new Error(errorMsg);
            }

            // Verify that parsed response is a valid JSON object
            if (!parsed.isJson || !parsed.data) {
                throw new Error("Server returned an invalid or empty response instead of JSON.");
            }

            const data = parsed.data;

            if (data.status !== "success") {
                throw new Error(data.message || "Music generation failed.");
            }

            console.log("[Generate] Success response:", data);

            // Populate Metadata Card
            metaFilename.textContent = data.filename || "-";
            metaDuration.textContent = `${data.duration_seconds || 0}s`;
            metaNotes.textContent = `${data.num_notes || numNotes} notes`;
            metaGenTime.textContent = `${data.generation_time_seconds || 0}s`;
            metadataCard.classList.remove("hidden");

            // Setup Download Button
            if (data.download_url) {
                downloadBtn.href = data.download_url;
                downloadBtn.setAttribute("download", data.filename || "generated_music.mid");
                downloadBtn.classList.remove("disabled");
                downloadBtnText.textContent = `Download ${data.filename || "MIDI"}`;
            }

            // Load Generated MIDI for In-Browser Playback
            if (data.download_url) {
                await loadMidiFile(data.download_url, data.duration_seconds);
            }

            playerBadge.textContent = "Ready to Play";
            playerBadge.style.color = "#10b981";

        } catch (err) {
            console.error("[Generate] Generation failed:", err);
            errorMessage.textContent = err.message || "Failed to communicate with music engine.";
            errorAlert.classList.remove("hidden");
            playerBadge.textContent = "Error";
            playerBadge.style.color = "#ef4444";
        } finally {
            // Guarantee that the Generate button is restored and spinner is hidden
            generateBtn.disabled = false;
            statusBanner.classList.add("hidden");
        }
    }

    // -------------------------------------------------------------------------
    // 4. In-Browser MIDI Loading & Parsing
    // -------------------------------------------------------------------------
    async function loadMidiFile(midiUrl, serverDuration) {
        overlayMessage.textContent = "Parsing MIDI for playback...";

        if (typeof Midi === "undefined") {
            overlayMessage.innerHTML = "⚠️ In-browser player requires Tone.js / @tonejs/midi.<br>You can still download the .mid file below!";
            console.warn("[Player] Midi parser library not loaded from CDN.");
            return;
        }

        try {
            initSynthesizer();

            // Load and parse MIDI with fallback handling
            let midi;
            try {
                midi = await Midi.fromUrl(midiUrl);
            } catch (urlErr) {
                console.warn("[Player] Midi.fromUrl failed, trying safe direct fetch:", urlErr);
                const midiRes = await fetch(midiUrl);
                if (!midiRes.ok) {
                    throw new Error(`Failed to load MIDI file from server (HTTP ${midiRes.status}).`);
                }
                const buffer = await midiRes.arrayBuffer();
                midi = new Midi(buffer);
            }
            currentMidiData = midi;

            // Extract all notes
            scheduledNotes = [];
            midi.tracks.forEach(track => {
                track.notes.forEach(n => {
                    scheduledNotes.push({
                        name: n.name,
                        midi: n.midi,
                        time: n.time,
                        duration: n.duration,
                        velocity: n.velocity,
                        played: false
                    });
                });
            });

            // Sort by start time
            scheduledNotes.sort((a, b) => a.time - b.time);

            // Set duration
            playbackDuration = midi.duration || serverDuration || 5.0;
            totalTimeLabel.textContent = formatTime(playbackDuration);
            currentTimeLabel.textContent = "0:00";
            seekSlider.max = playbackDuration;
            seekSlider.value = 0;
            seekSlider.disabled = false;

            // Enable player buttons
            playBtn.disabled = false;
            stopBtn.disabled = false;
            visualizerOverlay.classList.add("hidden");

            console.log(`[Player] Loaded ${scheduledNotes.length} notes, duration: ${playbackDuration.toFixed(2)}s`);
            drawVisualizerIdle();

        } catch (err) {
            console.error("[Player] Failed to parse MIDI:", err);
            overlayMessage.innerHTML = `⚠️ Playback error: ${err.message}<br>You can still download the MIDI file directly.`;
        }
    }

    // -------------------------------------------------------------------------
    // 5. Playback Engine
    // -------------------------------------------------------------------------
    playBtn.addEventListener("click", togglePlayPause);
    stopBtn.addEventListener("click", stopPlayback);

    async function togglePlayPause() {
        if (!scheduledNotes.length) return;

        if (isPlaying) {
            pausePlayback();
        } else {
            await startPlayback();
        }
    }

    async function startPlayback() {
        if (typeof Tone !== "undefined") {
            await Tone.start();
        }

        isPlaying = true;
        playIcon.classList.add("hidden");
        pauseIcon.classList.remove("hidden");
        playerBadge.textContent = "Playing";
        playerBadge.style.color = "#06b6d4";

        playbackStartTime = performance.now() - (pauseOffset * 1000);

        // Reset 'played' flags for notes ahead of pauseOffset
        scheduledNotes.forEach(n => {
            n.played = n.time < pauseOffset;
        });

        requestAnimationFrame(playbackLoop);
    }

    function pausePlayback() {
        isPlaying = false;
        playIcon.classList.remove("hidden");
        pauseIcon.classList.add("hidden");
        playerBadge.textContent = "Paused";
        playerBadge.style.color = "#f59e0b";

        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }

        // Release all active synth voices
        if (toneSynth) {
            toneSynth.releaseAll();
        }
        activeVisualizerNotes = [];
    }

    function stopPlayback() {
        pausePlayback();
        pauseOffset = 0;
        seekSlider.value = 0;
        currentTimeLabel.textContent = "0:00";
        playerBadge.textContent = scheduledNotes.length ? "Stopped" : "Ready";
        playerBadge.style.color = "#94a3b8";

        scheduledNotes.forEach(n => {
            n.played = false;
        });

        drawVisualizerIdle();
    }

    function playbackLoop() {
        if (!isPlaying) return;

        const now = performance.now();
        const elapsedSec = (now - playbackStartTime) / 1000;
        pauseOffset = elapsedSec;

        // Update seek slider & timer
        seekSlider.value = Math.min(elapsedSec, playbackDuration);
        currentTimeLabel.textContent = formatTime(elapsedSec);

        // Trigger notes whose time has arrived
        const lookahead = 0.05; // 50ms buffer
        scheduledNotes.forEach(note => {
            if (!note.played && note.time <= elapsedSec + lookahead) {
                note.played = true;

                // Play synth note
                if (toneSynth) {
                    try {
                        const noteVel = Math.max(0.1, Math.min(note.velocity, 1.0));
                        toneSynth.triggerAttackRelease(note.name, note.duration, undefined, noteVel);
                    } catch (e) {
                        // ignore voice limit errors
                    }
                }

                // Add to active visualizer queue
                activeVisualizerNotes.push({
                    midi: note.midi,
                    name: note.name,
                    startTime: performance.now(),
                    duration: note.duration * 1000
                });
            }
        });

        // Clean up expired visualizer notes
        const currentMs = performance.now();
        activeVisualizerNotes = activeVisualizerNotes.filter(n => (currentMs - n.startTime) < n.duration);

        // Render dynamic visualizer
        drawVisualizerPlaying(elapsedSec);

        // Check if finished
        if (elapsedSec >= playbackDuration + 0.5) {
            stopPlayback();
            return;
        }

        animationFrameId = requestAnimationFrame(playbackLoop);
    }

    // Seek Bar Interaction
    seekSlider.addEventListener("input", () => {
        if (!scheduledNotes.length) return;
        const targetSec = parseFloat(seekSlider.value);
        pauseOffset = targetSec;
        currentTimeLabel.textContent = formatTime(targetSec);

        if (toneSynth) {
            toneSynth.releaseAll();
        }

        // Reset note played state according to scrub position
        scheduledNotes.forEach(n => {
            n.played = n.time < targetSec;
        });

        if (isPlaying) {
            playbackStartTime = performance.now() - (targetSec * 1000);
        }
    });

    // Volume Slider
    volumeSlider.addEventListener("input", () => {
        const vol = parseFloat(volumeSlider.value);
        if (typeof Tone !== "undefined" && Tone.Destination) {
            if (vol === 0) {
                Tone.Destination.mute = true;
            } else {
                Tone.Destination.mute = false;
                Tone.Destination.volume.value = Tone.gainToDb(vol);
            }
        }
    });

    // -------------------------------------------------------------------------
    // 6. Interactive Visualizer (Canvas)
    // -------------------------------------------------------------------------
    const ctx = visualizerCanvas.getContext("2d");

    function resizeCanvas() {
        const rect = visualizerCanvas.getBoundingClientRect();
        visualizerCanvas.width = rect.width * window.devicePixelRatio;
        visualizerCanvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    window.addEventListener("resize", () => {
        resizeCanvas();
        if (!isPlaying) drawVisualizerIdle();
    });
    resizeCanvas();

    function drawVisualizerIdle() {
        const width = visualizerCanvas.clientWidth;
        const height = visualizerCanvas.clientHeight;
        ctx.clearRect(0, 0, width, height);

        // Subtle resting frequency bars
        const numBars = 32;
        const barWidth = (width / numBars) - 3;

        for (let i = 0; i < numBars; i++) {
            const barHeight = 8 + Math.sin(i * 0.4) * 4;
            const x = i * (barWidth + 3) + 2;
            const y = height - barHeight - 10;

            const grad = ctx.createLinearGradient(0, y, 0, height);
            grad.addColorStop(0, "rgba(139, 92, 246, 0.4)");
            grad.addColorStop(1, "rgba(6, 182, 212, 0.1)");

            ctx.fillStyle = grad;
            ctx.fillRect(x, y, barWidth, barHeight);
        }
    }

    function drawVisualizerPlaying(elapsedSec) {
        const width = visualizerCanvas.clientWidth;
        const height = visualizerCanvas.clientHeight;
        ctx.clearRect(0, 0, width, height);

        // Background grid lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 1;
        for (let y = 20; y < height; y += 30) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw note bars
        const numBars = 48;
        const barWidth = (width / numBars) - 2;

        for (let i = 0; i < numBars; i++) {
            // Map bar index to MIDI pitch range (25 to 99)
            const approxMidi = 25 + Math.round((i / numBars) * (99 - 25));

            // Check if this pitch is active
            const isActive = activeVisualizerNotes.some(n => Math.abs(n.midi - approxMidi) <= 1);

            let barHeight = 6;
            if (isActive) {
                barHeight = height * 0.75 + Math.random() * 15;
            } else {
                // Background rhythmic ripple
                barHeight = 6 + Math.abs(Math.sin((i * 0.3) + elapsedSec * 4)) * 12;
            }

            const x = i * (barWidth + 2) + 1;
            const y = height - barHeight - 6;

            const grad = ctx.createLinearGradient(0, y, 0, height);
            if (isActive) {
                grad.addColorStop(0, "#06b6d4");
                grad.addColorStop(0.5, "#8b5cf6");
                grad.addColorStop(1, "rgba(139, 92, 246, 0.2)");
            } else {
                grad.addColorStop(0, "rgba(139, 92, 246, 0.35)");
                grad.addColorStop(1, "rgba(14, 18, 34, 0.2)");
            }

            ctx.fillStyle = grad;
            ctx.shadowBlur = isActive ? 12 : 0;
            ctx.shadowColor = isActive ? "#06b6d4" : "transparent";
            ctx.fillRect(x, y, barWidth, barHeight);
            ctx.shadowBlur = 0;
        }
    }

    // Helper: Format seconds to M:SS
    function formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return "0:00";
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
    }

    // Draw initial idle visualizer
    drawVisualizerIdle();
});
