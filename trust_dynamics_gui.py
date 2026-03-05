# trust_web_multi_v2.py
"""
Trust Dynamics - Multi-Simulation Web Interface (Fixed)

Features:
- 4 simulations in 2x2 grid (MAIN SCREEN)
- Sidebar with controls (LEFT SIDE)
- Pause All / Resume All
- Pause This / Resume This
- Full stats panel when simulation selected
"""

from flask import Flask, jsonify, request
import multiprocessing as mp
from multi_sim_engine import (
    MultiSimEngine, 
    get_mobility_comparison,
    get_trust_threshold_comparison,
    get_initial_trust_comparison
)


app = Flask(__name__)

engine = None
current_scenario = 'mobility'

SCENARIOS = {
    'mobility': get_mobility_comparison,
    'threshold': get_trust_threshold_comparison,
    'initial_trust': get_initial_trust_comparison,
}


@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trust Dynamics Multi-Sim</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e8e8e8;
            height: 100vh;
            display: flex;
            overflow: hidden;
        }
        
        /* LEFT SIDEBAR */
        #sidebar {
            width: 300px;
            background: #111;
            border-right: 1px solid #222;
            display: flex;
            flex-direction: column;
        }
        
        #header {
            padding: 20px;
            border-bottom: 1px solid #222;
        }
        
        h1 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 4px;
            color: #fff;
        }
        
        .subtitle { font-size: 11px; color: #666; }
        
        #scenario-selector {
            padding: 16px 20px;
            border-bottom: 1px solid #222;
        }
        
        select {
            width: 100%;
            padding: 10px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            color: #e8e8e8;
            font-size: 13px;
            cursor: pointer;
        }
        
        #controls {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            -webkit-overflow-scrolling: touch;
        }
        
        .selected-sim {
            background: #1a1a1a;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 16px;
            border: 1px solid #333;
        }
        
        .selected-sim-name {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        
        .selected-sim-title {
            font-size: 15px;
            color: #fff;
            font-weight: 500;
        }
        
        .buttons {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }
        
        button {
            flex: 1;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #1a1a1a;
            color: #e8e8e8;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            -webkit-tap-highlight-color: transparent;
        }
        
        button:hover { background: #222; border-color: #444; }
        button:active { transform: scale(0.98); }
        
        .section-divider {
            height: 1px;
            background: #222;
            margin: 16px 0;
        }
        
        .stats-panel {
            background: #1a1a1a;
            padding: 14px;
            border-radius: 6px;
            border: 1px solid #222;
            font-family: Monaco, monospace;
            font-size: 11px;
            line-height: 2;
            color: #999;
        }
        
        .stats-panel .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2px;
        }
        
        .stats-panel .stat-value {
            color: #e8e8e8;
            font-weight: 500;
        }
        
        .stats-section {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #222;
        }
        
        .stats-section:first-child {
            margin-top: 0;
            padding-top: 0;
            border-top: none;
        }
        
        .stats-section-title {
            color: #666;
            font-size: 10px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .slider-group {
            margin-bottom: 16px;
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            margin-bottom: 6px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .slider-label span:last-child {
            color: #e8e8e8;
            font-family: Monaco, monospace;
            font-size: 11px;
            text-transform: none;
        }
        
        input[type="range"] {
            width: 100%;
            height: 4px;
            background: #222;
            outline: none;
            border-radius: 2px;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #e8e8e8;
            cursor: pointer;
            border: 3px solid #0a0a0a;
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #e8e8e8;
            cursor: pointer;
            border: 3px solid #0a0a0a;
        }
        
        input[type="range"]:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* RIGHT SIDE - MAIN GRID */
        #grid-container {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 1px;
            background: #000;
            padding: 1px;
        }
        
        #grid-container.maximized {
            grid-template-columns: 1fr;
            grid-template-rows: 1fr;
        }
        
        .sim-panel {
            background: #111;
            position: relative;
            cursor: pointer;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        #grid-container.maximized .sim-panel { display: none; }
        #grid-container.maximized .sim-panel.maximized { display: flex; }
        
        .sim-panel.selected {
            outline: 2px solid #4a90e2;
            outline-offset: -2px;
        }
        
        .sim-header {
            padding: 12px 16px;
            background: rgba(0,0,0,0.3);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .sim-title {
            font-size: 13px;
            font-weight: 500;
            color: #e8e8e8;
        }
        
        .sim-round {
            font-size: 11px;
            color: #666;
            font-family: Monaco, monospace;
        }
        
        .sim-canvas-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 12px;
            gap: 12px;
        }
        
        canvas {
            background: #0a0a0a;
            border-radius: 4px;
        }
        
        .grid-canvas { flex: 2; }
        .chart-canvas { flex: 1; }
        
        .maximize-btn {
            position: absolute;
            top: 12px;
            right: 16px;
            background: rgba(0,0,0,0.5);
            border: 1px solid #333;
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 11px;
            color: #e8e8e8;
            cursor: pointer;
            z-index: 10;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .sim-panel:hover .maximize-btn { opacity: 1; }
        .maximize-btn:hover {
            background: rgba(0,0,0,0.7);
            border-color: #555;
        }
    </style>
</head>
<body>
    <!-- LEFT SIDEBAR -->
    <div id="sidebar">
        <div id="header">
            <h1>Multi-Simulation Explorer</h1>
            <div class="subtitle">4 Parallel Simulations</div>
        </div>
        
        <div id="scenario-selector">
            <select id="scenarioSelect" onchange="changeScenario(this.value)">
                <option value="mobility">Mobility Comparison</option>
                <option value="threshold">Trust Threshold Comparison</option>
                <option value="initial_trust">Initial Trust Comparison</option>
            </select>
        </div>
        
        <div id="controls">
            <div class="selected-sim">
                <div class="selected-sim-name">Currently Selected</div>
                <div class="selected-sim-title" id="selectedSimName">Click a simulation</div>
            </div>
            
            <!-- Global Controls -->
            <div class="buttons">
                <button id="pauseAllBtn" onclick="pauseAll()">⏸ Pause All</button>
                <button id="resumeAllBtn" onclick="resumeAll()" style="display:none;">▶ Resume All</button>
            </div>
            
            <div class="buttons">
                <button onclick="restartAll()">↻ Restart All</button>
            </div>
            
            <!-- Individual Controls -->
            <div class="buttons" id="individualControls" style="display:none;">
                <button id="pauseThisBtn" onclick="pauseThis()">⏸ Pause This</button>
                <button id="resumeThisBtn" onclick="resumeThis()" style="display:none;">▶ Resume This</button>
            </div>
            
            <div class="section-divider"></div>
            
            <!-- Full Stats Panel -->
            <div id="fullStats" style="display:none;">
                <div class="stats-panel">
                    <div class="stats-section">
                        <div class="stats-section-title">Round</div>
                        <div class="stat-row">
                            <span>Current</span>
                            <span class="stat-value" id="full_round">-</span>
                        </div>
                    </div>
                    
                    <div class="stats-section">
                        <div class="stats-section-title">Parameters</div>
                        <div class="stat-row">
                            <span>Mobility</span>
                            <span class="stat-value" id="full_mobility">-</span>
                        </div>
                        <div class="stat-row">
                            <span>Threshold</span>
                            <span class="stat-value" id="full_threshold">-</span>
                        </div>
                        <div class="stat-row">
                            <span>Init Trust</span>
                            <span class="stat-value" id="full_init_trust">-</span>
                        </div>
                        <div class="stat-row">
                            <span>% Trustworthy</span>
                            <span class="stat-value" id="full_share_trust">-</span>
                        </div>
                        <div class="stat-row">
                            <span>Learn Rate</span>
                            <span class="stat-value" id="full_sensitivity">-</span>
                        </div>
                    </div>
                    
                    <div class="stats-section">
                        <div class="stats-section-title">Current State</div>
                        <div class="stat-row">
                            <span>Avg Trust</span>
                            <span class="stat-value" id="full_avg_trust">-</span>
                        </div>
                        <div class="stat-row">
                            <span>% Trusting</span>
                            <span class="stat-value" id="full_pct_trust">-</span>
                        </div>
                        <div class="stat-row">
                            <span>Coop Rate</span>
                            <span class="stat-value" id="full_coop">-</span>
                        </div>
                    </div>
                    
                    <div class="stats-section">
                        <div class="stats-section-title">Analysis</div>
                        <div class="stat-row">
                            <span>Trend</span>
                            <span class="stat-value" id="full_trend">-</span>
                        </div>
                        <div class="stat-row">
                            <span>State</span>
                            <span class="stat-value" id="full_equilibrium">-</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section-divider"></div>
            
            <!-- Parameter Controls -->
            <div style="margin-top: 16px;">
                <div style="font-size: 12px; color: #999; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Adjust Parameters
                </div>
                
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Mobility</span><span id="v_mobility">-</span>
                    </div>
                    <input type="range" id="mobility" min="1" max="20" value="5" 
                           oninput="updateParam('mobility', this.value)" disabled>
                </div>
                
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Trust Threshold</span><span id="v_trust_threshold">-</span>
                    </div>
                    <input type="range" id="trust_threshold" min="0" max="1" step="0.05" value="0.5" 
                           oninput="updateParam('trust_threshold', this.value)" disabled>
                </div>
                
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Initial Trust</span><span id="v_initial_trust_mean">-</span>
                    </div>
                    <input type="range" id="initial_trust_mean" min="0" max="1" step="0.05" value="0.5" 
                           oninput="updateParam('initial_trust_mean', this.value)" disabled>
                </div>
                
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Share Trustworthy</span><span id="v_share_trustworthy">-</span>
                    </div>
                    <input type="range" id="share_trustworthy" min="0" max="1" step="0.05" value="0.6" 
                           oninput="updateParam('share_trustworthy', this.value)" disabled>
                </div>
                
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Learning Rate</span><span id="v_sensitivity">-</span>
                    </div>
                    <input type="range" id="sensitivity" min="0.01" max="0.2" step="0.01" value="0.05" 
                           oninput="updateParam('sensitivity', this.value)" disabled>
                </div>
            </div>
        </div>
    </div>
    
    <!-- RIGHT SIDE - 2x2 GRID -->
    <div id="grid-container">
        <div class="sim-panel" data-sim-id="0" onclick="selectSim(0)">
            <button class="maximize-btn" onclick="event.stopPropagation(); toggleMaximize(0)">⛶</button>
            <div class="sim-header">
                <div class="sim-title" id="title-0">Simulation 1</div>
                <div class="sim-round" id="round-0">Round 0</div>
            </div>
            <div class="sim-canvas-container">
                <canvas class="grid-canvas" id="grid-0"></canvas>
                <canvas class="chart-canvas" id="chart-0"></canvas>
            </div>
        </div>
        
        <div class="sim-panel" data-sim-id="1" onclick="selectSim(1)">
            <button class="maximize-btn" onclick="event.stopPropagation(); toggleMaximize(1)">⛶</button>
            <div class="sim-header">
                <div class="sim-title" id="title-1">Simulation 2</div>
                <div class="sim-round" id="round-1">Round 0</div>
            </div>
            <div class="sim-canvas-container">
                <canvas class="grid-canvas" id="grid-1"></canvas>
                <canvas class="chart-canvas" id="chart-1"></canvas>
            </div>
        </div>
        
        <div class="sim-panel" data-sim-id="2" onclick="selectSim(2)">
            <button class="maximize-btn" onclick="event.stopPropagation(); toggleMaximize(2)">⛶</button>
            <div class="sim-header">
                <div class="sim-title" id="title-2">Simulation 3</div>
                <div class="sim-round" id="round-2">Round 0</div>
            </div>
            <div class="sim-canvas-container">
                <canvas class="grid-canvas" id="grid-2"></canvas>
                <canvas class="chart-canvas" id="chart-2"></canvas>
            </div>
        </div>
        
        <div class="sim-panel" data-sim-id="3" onclick="selectSim(3)">
            <button class="maximize-btn" onclick="event.stopPropagation(); toggleMaximize(3)">⛶</button>
            <div class="sim-header">
                <div class="sim-title" id="title-3">Simulation 4</div>
                <div class="sim-round" id="round-3">Round 0</div>
            </div>
            <div class="sim-canvas-container">
                <canvas class="grid-canvas" id="grid-3"></canvas>
                <canvas class="chart-canvas" id="chart-3"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        let selectedSimId = null;
        let maximizedSimId = null;
        let simParams = {};
        let pausedSims = new Set();
        let allPaused = false;
        
        // Initialize params for each sim
        simParams = {
            0: {mobility: 1, trust_threshold: 0.5, initial_trust_mean: 0.5, 
                share_trustworthy: 0.6, sensitivity: 0.05},
            1: {mobility: 5, trust_threshold: 0.5, initial_trust_mean: 0.5, 
                share_trustworthy: 0.6, sensitivity: 0.05},
            2: {mobility: 10, trust_threshold: 0.5, initial_trust_mean: 0.5, 
                share_trustworthy: 0.6, sensitivity: 0.05},
            3: {mobility: 20, trust_threshold: 0.5, initial_trust_mean: 0.5, 
                share_trustworthy: 0.6, sensitivity: 0.05}
        };
        
        function resizeCanvases() {
            for (let i = 0; i < 4; i++) {
                const gridCanvas = document.getElementById(`grid-${i}`);
                const chartCanvas = document.getElementById(`chart-${i}`);
                
                gridCanvas.width = gridCanvas.offsetWidth;
                gridCanvas.height = gridCanvas.offsetHeight;
                chartCanvas.width = chartCanvas.offsetWidth;
                chartCanvas.height = chartCanvas.offsetHeight;
            }
        }
        
        window.addEventListener('resize', resizeCanvases);
        setTimeout(resizeCanvases, 100);
        
        function selectSim(simId) {
            selectedSimId = simId;
            
            document.querySelectorAll('.sim-panel').forEach(panel => {
                panel.classList.remove('selected');
            });
            document.querySelector(`[data-sim-id="${simId}"]`).classList.add('selected');
            
            const simName = document.getElementById(`title-${simId}`).textContent;
            document.getElementById('selectedSimName').textContent = simName;
            
            document.getElementById('individualControls').style.display = 'flex';
            document.getElementById('fullStats').style.display = 'block';
            
            // Enable sliders
            document.querySelectorAll('input[type="range"]').forEach(slider => {
                slider.disabled = false;
            });
            
            // Load current params for this sim
            const params = simParams[simId];
            updateSliderDisplay('mobility', params.mobility);
            updateSliderDisplay('trust_threshold', params.trust_threshold);
            updateSliderDisplay('initial_trust_mean', params.initial_trust_mean);
            updateSliderDisplay('share_trustworthy', params.share_trustworthy);
            updateSliderDisplay('sensitivity', params.sensitivity);
            
            // Update pause/resume button for this sim
            if (pausedSims.has(simId)) {
                document.getElementById('pauseThisBtn').style.display = 'none';
                document.getElementById('resumeThisBtn').style.display = 'block';
            } else {
                document.getElementById('pauseThisBtn').style.display = 'block';
                document.getElementById('resumeThisBtn').style.display = 'none';
            }
        }
        
        function updateSliderDisplay(name, value) {
            const slider = document.getElementById(name);
            const display = document.getElementById(`v_${name}`);
            
            slider.value = value;
            const v = parseFloat(value);
            display.textContent = v % 1 === 0 ? v : v.toFixed(2);
        }
        
        function updateParam(name, value) {
            if (selectedSimId === null) return;
            
            const v = parseFloat(value);
            updateSliderDisplay(name, v);
            
            // Update stored params
            simParams[selectedSimId][name] = v;
            
            // Send update to backend
            fetch('/api/update_param', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sim_id: selectedSimId,
                    param: name,
                    value: v
                })
            });
            
            // Update the simulation name to reflect new param
            updateSimName(selectedSimId);
        }
        
        function updateSimName(simId) {
            const params = simParams[simId];
            let name = '';
            
            // Generate name based on what varies
            const scenario = document.getElementById('scenarioSelect').value;
            
            if (scenario === 'mobility') {
                name = `Mobility ${params.mobility}`;
            } else if (scenario === 'threshold') {
                name = `Threshold ${params.trust_threshold.toFixed(2)}`;
            } else if (scenario === 'initial_trust') {
                name = `Init Trust ${params.initial_trust_mean.toFixed(2)}`;
            }
            
            document.getElementById(`title-${simId}`).textContent = name;
        }
        
        function toggleMaximize(simId) {
            const container = document.getElementById('grid-container');
            const panel = document.querySelector(`[data-sim-id="${simId}"]`);
            
            if (maximizedSimId === simId) {
                container.classList.remove('maximized');
                panel.classList.remove('maximized');
                maximizedSimId = null;
            } else {
                container.classList.add('maximized');
                if (maximizedSimId !== null) {
                    document.querySelector(`[data-sim-id="${maximizedSimId}"]`).classList.remove('maximized');
                }
                panel.classList.add('maximized');
                maximizedSimId = simId;
            }
            
            setTimeout(resizeCanvases, 100);
        }
        
        function pauseAll() {
            allPaused = true;
            document.getElementById('pauseAllBtn').style.display = 'none';
            document.getElementById('resumeAllBtn').style.display = 'block';
            fetch('/api/pause', {method: 'POST'});
        }
        
        function resumeAll() {
            allPaused = false;
            pausedSims.clear();
            document.getElementById('pauseAllBtn').style.display = 'block';
            document.getElementById('resumeAllBtn').style.display = 'none';
            fetch('/api/resume', {method: 'POST'});
            
            if (selectedSimId !== null) {
                document.getElementById('pauseThisBtn').style.display = 'block';
                document.getElementById('resumeThisBtn').style.display = 'none';
            }
        }
        
        function pauseThis() {
            if (selectedSimId === null) return;
            pausedSims.add(selectedSimId);
            document.getElementById('pauseThisBtn').style.display = 'none';
            document.getElementById('resumeThisBtn').style.display = 'block';
            
            fetch('/api/pause_sim', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sim_id: selectedSimId})
            });
        }
        
        function resumeThis() {
            if (selectedSimId === null) return;
            pausedSims.delete(selectedSimId);
            document.getElementById('pauseThisBtn').style.display = 'block';
            document.getElementById('resumeThisBtn').style.display = 'none';
            
            fetch('/api/resume_sim', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sim_id: selectedSimId})
            });
        }
        
        function restartAll() {
            // Send current params to backend for restart
            fetch('/api/restart', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    params: simParams
                })
            });
        }
        
        function changeScenario(scenario) {
            // Update params based on scenario but preserve custom changes
            if (scenario === 'mobility') {
                simParams[0].mobility = 1;
                simParams[1].mobility = 5;
                simParams[2].mobility = 10;
                simParams[3].mobility = 20;
            } else if (scenario === 'threshold') {
                simParams[0].trust_threshold = 0.3;
                simParams[1].trust_threshold = 0.5;
                simParams[2].trust_threshold = 0.7;
                simParams[3].trust_threshold = 0.9;
            } else if (scenario === 'initial_trust') {
                simParams[0].initial_trust_mean = 0.3;
                simParams[1].initial_trust_mean = 0.5;
                simParams[2].initial_trust_mean = 0.7;
                simParams[3].initial_trust_mean = 0.9;
            }
            
            fetch('/api/change_scenario', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    scenario: scenario,
                    params: simParams
                })
            });
            
            // Update all sim names
            for (let i = 0; i < 4; i++) {
                updateSimName(i);
            }
        }
        
        function getTrend(history) {
            if (history.length < 10) return 'Gathering data';
            
            const recent = history.slice(-10);
            const avg_recent = recent.reduce((a, b) => a + b, 0) / recent.length;
            const earlier = history.slice(-20, -10);
            const avg_earlier = earlier.reduce((a, b) => a + b, 0) / earlier.length;
            
            const change = avg_recent - avg_earlier;
            
            if (Math.abs(change) < 0.01) return '→ Stable';
            if (change > 0.05) return '↑ Rising Fast';
            if (change > 0) return '↗ Rising';
            if (change < -0.05) return '↓ Falling Fast';
            return '↘ Falling';
        }
        
        function getEquilibrium(history) {
            if (history.length < 20) return 'Developing';
            
            const recent = history.slice(-20);
            const mean = recent.reduce((a, b) => a + b) / recent.length;
            const variance = recent.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / recent.length;
            
            if (variance < 0.001) return 'Equilibrium';
            if (variance < 0.01) return 'Near Equilibrium';
            return 'Evolving';
        }
        
        function updateFullStats(state, params) {
            if (!state || !params) return;
            
            document.getElementById('full_round').textContent = state.round;
            document.getElementById('full_mobility').textContent = params.mobility || '-';
            document.getElementById('full_threshold').textContent = (params.trust_threshold || 0).toFixed(2);
            document.getElementById('full_init_trust').textContent = (params.initial_trust_mean || 0).toFixed(2);
            document.getElementById('full_share_trust').textContent = 
                ((params.share_trustworthy || 0) * 100).toFixed(0) + '%';
            document.getElementById('full_sensitivity').textContent = (params.sensitivity || 0).toFixed(3);
            
            document.getElementById('full_avg_trust').textContent = state.stats.avg_trust.toFixed(3);
            document.getElementById('full_pct_trust').textContent = state.stats.pct_trusting.toFixed(1) + '%';
            document.getElementById('full_coop').textContent = state.stats.coop_rate.toFixed(3);
            
            document.getElementById('full_trend').textContent = getTrend(state.trust_history);
            document.getElementById('full_equilibrium').textContent = getEquilibrium(state.trust_history);
        }
        
        function color(t) {
            if (t < 0.33) return '#f44336';
            if (t < 0.66) return '#ffeb3b';
            return '#4caf50';
        }
        
        function drawGrid(canvasId, agents) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const cell = Math.min(w, h) / 51;
            
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, w, h);
            
            // Draw subtle grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 51; i++) {
                ctx.beginPath();
                ctx.moveTo(i * cell, 0);
                ctx.lineTo(i * cell, 51 * cell);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, i * cell);
                ctx.lineTo(51 * cell, i * cell);
                ctx.stroke();
            }
            
            // First pass: Draw all agents (background layer)
            agents.forEach(a => {
                const x = a.x * cell + cell/2;
                const y = a.y * cell + cell/2;
                const r = 3;
                
                // Outer glow
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
                gradient.addColorStop(0, color(a.t) + '40');
                gradient.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, r * 2, 0, Math.PI * 2);
                ctx.fill();
                
                // Main dot
                ctx.fillStyle = color(a.t);
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fill();
                
                // Border to distinguish agents
                ctx.strokeStyle = '#0a0a0a';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }
        
        function drawChart(canvasId, history) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const pad = 20;
            
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, w, h);
            
            if (history.length < 2) return;
            
            const pw = w - 2*pad, ph = h - 2*pad;
            
            ctx.strokeStyle = '#2196f3';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            history.forEach((v, i) => {
                const x = pad + (i / (history.length-1)) * pw;
                const y = h - pad - v * ph;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        function updateSimulation(simId, state) {
            document.getElementById(`title-${simId}`).textContent = state.name;
            document.getElementById(`round-${simId}`).textContent = `Round ${state.round}`;
            
            drawGrid(`grid-${simId}`, state.agents, state.interactions);
            drawChart(`chart-${simId}`, state.trust_history);
            
            if (selectedSimId === simId) {
                updateFullStats(state, simParams[simId]);
            }
        }
        
        function drawGrid(canvasId, agents, interactions) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const cell = Math.min(w, h) / 51;
            
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, w, h);
            
            // Draw subtle grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 51; i++) {
                ctx.beginPath();
                ctx.moveTo(i * cell, 0);
                ctx.lineTo(i * cell, 51 * cell);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, i * cell);
                ctx.lineTo(51 * cell, i * cell);
                ctx.stroke();
            }
            
            // Draw interaction lines FIRST (behind agents)
            if (interactions && interactions.length > 0) {
                interactions.forEach(inter => {
                    const x1 = inter.trustor_pos[0] * cell + cell/2;
                    const y1 = inter.trustor_pos[1] * cell + cell/2;
                    const x2 = inter.trustee_pos[0] * cell + cell/2;
                    const y2 = inter.trustee_pos[1] * cell + cell/2;
                    
                    const interColor = inter.cooperated ? '#4caf50' : '#f44336';
                    
                    // Glowing line effect
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = interColor;
                    
                    // Main line
                    ctx.strokeStyle = interColor;
                    ctx.lineWidth = inter.cooperated ? 2.5 : 1.5;
                    ctx.setLineDash(inter.cooperated ? [] : [4, 4]);
                    ctx.globalAlpha = 0.8;
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                    
                    ctx.globalAlpha = 1.0;
                    ctx.shadowBlur = 0;
                    ctx.setLineDash([]);
                    
                    // Draw X mark at interaction midpoint for betrayal
                    if (!inter.cooperated) {
                        const mx = (x1 + x2) / 2;
                        const my = (y1 + y2) / 2;
                        const size = 4;
                        
                        ctx.strokeStyle = '#ff5252';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(mx - size, my - size);
                        ctx.lineTo(mx + size, my + size);
                        ctx.moveTo(mx + size, my - size);
                        ctx.lineTo(mx - size, my + size);
                        ctx.stroke();
                    }
                    
                    // Draw circle at interaction midpoint for cooperation
                    if (inter.cooperated) {
                        const mx = (x1 + x2) / 2;
                        const my = (y1 + y2) / 2;
                        
                        ctx.fillStyle = interColor + '80';
                        ctx.beginPath();
                        ctx.arc(mx, my, 3, 0, Math.PI * 2);
                        ctx.fill();
                    }
                });
            }
            
            // Draw agents (foreground layer)
            agents.forEach(a => {
                const x = a.x * cell + cell/2;
                const y = a.y * cell + cell/2;
                const r = 3.5;
                
                // Outer glow
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, r * 2.5);
                gradient.addColorStop(0, color(a.t) + '60');
                gradient.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
                ctx.fill();
                
                // Main dot
                ctx.fillStyle = color(a.t);
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fill();
                
                // Border
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 1.5;
                ctx.stroke();
                
                // Inner highlight
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.beginPath();
                ctx.arc(x - r/3, y - r/3, r/3, 0, Math.PI * 2);
                ctx.fill();
            });
        }
        
        async function updateAll() {
            try {
                const response = await fetch('/api/states');
                const states = await response.json();
                
                for (let simId in states) {
                    updateSimulation(parseInt(simId), states[simId]);
                }
            } catch (error) {
                console.error('Update error:', error);
            }
            
            // Use requestAnimationFrame for smoother updates
            requestAnimationFrame(updateAll);
        }
        
        fetch('/api/init', {method: 'POST'}).then(() => {
            setTimeout(resizeCanvases, 200);
            requestAnimationFrame(updateAll);
        });
    </script>
</body>
</html>
    """


@app.route('/api/init', methods=['POST'])
def api_init():
    global engine, current_scenario
    scenarios = SCENARIOS[current_scenario]()
    engine = MultiSimEngine(scenarios, max_workers=4)
    engine.start()
    return jsonify({'status': 'ok'})


@app.route('/api/states', methods=['GET'])
def api_states():
    if engine:
        return jsonify(engine.get_all_states())
    return jsonify({})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    if engine:
        engine.pause()
    return jsonify({'status': 'ok'})


@app.route('/api/resume', methods=['POST'])
def api_resume():
    if engine:
        engine.resume()
    return jsonify({'status': 'ok'})


@app.route('/api/pause_sim', methods=['POST'])
def api_pause_sim():
    if engine:
        data = request.json
        sim_id = data.get('sim_id')
        engine.pause(sim_id)
    return jsonify({'status': 'ok'})


@app.route('/api/resume_sim', methods=['POST'])
def api_resume_sim():
    if engine:
        data = request.json
        sim_id = data.get('sim_id')
        engine.resume(sim_id)
    return jsonify({'status': 'ok'})


@app.route('/api/restart', methods=['POST'])
def api_restart():
    global engine
    
    # Get custom params if provided
    data = request.json or {}
    custom_params = data.get('params')
    
    if engine:
        engine.stop()
    
    # Use custom params if provided, otherwise default
    if custom_params:
        scenarios = []
        for sim_id in range(4):
            params_dict = custom_params.get(str(sim_id), custom_params.get(sim_id, {}))
            
            # Build scenario with custom params
            base_params = {
                'grid_size': 51,
                'n_agents': 600,
                'share_trustworthy': params_dict.get('share_trustworthy', 0.6),
                'initial_trust_mean': params_dict.get('initial_trust_mean', 0.5),
                'initial_trust_std': 0.2,
                'sensitivity': params_dict.get('sensitivity', 0.05),
                'trust_threshold': params_dict.get('trust_threshold', 0.5),
                'mobility': int(params_dict.get('mobility', 5)),
            }
            
            scenarios.append({
                'name': f'Sim {sim_id + 1}',
                'params': base_params,
                'seed': 42
            })
        
        engine = MultiSimEngine(scenarios, max_workers=4)
    else:
        api_init()
        return jsonify({'status': 'ok'})
    
    engine.start()
    return jsonify({'status': 'ok'})


@app.route('/api/change_scenario', methods=['POST'])
def api_change_scenario():
    global engine, current_scenario
    data = request.json
    new_scenario = data.get('scenario', 'mobility')
    custom_params = data.get('params')
    
    if new_scenario in SCENARIOS:
        current_scenario = new_scenario
        if engine:
            engine.stop()
        
        # Use custom params if provided
        if custom_params:
            scenarios = []
            for sim_id in range(4):
                params_dict = custom_params.get(str(sim_id), custom_params.get(sim_id, {}))
                
                base_params = {
                    'grid_size': 51,
                    'n_agents': 600,
                    'share_trustworthy': params_dict.get('share_trustworthy', 0.6),
                    'initial_trust_mean': params_dict.get('initial_trust_mean', 0.5),
                    'initial_trust_std': 0.2,
                    'sensitivity': params_dict.get('sensitivity', 0.05),
                    'trust_threshold': params_dict.get('trust_threshold', 0.5),
                    'mobility': int(params_dict.get('mobility', 5)),
                }
                
                scenarios.append({
                    'name': f'Sim {sim_id + 1}',
                    'params': base_params,
                    'seed': 42
                })
            
            engine = MultiSimEngine(scenarios, max_workers=4)
            engine.start()
        else:
            api_init()
    
    return jsonify({'status': 'ok'})


@app.route('/api/update_param', methods=['POST'])
def api_update_param():
    # Note: In current architecture, can't update running sim params
    # Would need to restart that specific simulation
    # For now, just acknowledge - params will apply on restart
    return jsonify({'status': 'ok'})


if __name__ == "__main__":
    import socket
    
    print("\n" + "="*60)
    print("MULTI-SIMULATION EXPLORER v2")
    print("="*60)
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n🌐 Access: http://{local_ip}:5000")
    print(f"🚀 Running {mp.cpu_count()} CPU cores available")
    print(f"⚡ 4 simulations in parallel")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(debug=False, port=5001, host='0.0.0.0', threaded=True)
    finally:
        if engine:
            engine.stop()