"""Duxx AI Studio — FastAPI-based web UI for agent management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Duxx AI Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --navy: #0c1b3a;
            --navy-light: #14284b;
            --pink: #d6336c;
            --pink-light: #f06595;
            --pink-bg: #fdf2f6;
            --black: #1a1a1a;
            --gray-900: #2d2d2d;
            --gray-700: #555;
            --gray-500: #888;
            --gray-300: #ccc;
            --gray-200: #e5e7eb;
            --gray-100: #f3f4f6;
            --gray-50: #f9fafb;
            --white: #ffffff;
            --green: #16a34a;
            --green-bg: #f0fdf4;
            --blue: #2563eb;
            --blue-bg: #eff6ff;
            --radius: 10px;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--gray-50); color: var(--black); }

        /* ── Header ── */
        .header {
            background: var(--white);
            padding: 0 32px;
            height: 64px;
            border-bottom: 1px solid var(--gray-200);
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-sm);
            position: relative;
            z-index: 10;
        }
        .header-brand { display: flex; align-items: center; gap: 10px; }
        .header-brand .logo {
            width: 34px; height: 34px; border-radius: 8px;
            background: linear-gradient(135deg, var(--navy), var(--pink));
            display: flex; align-items: center; justify-content: center;
            color: var(--white); font-weight: 700; font-size: 14px;
        }
        .header h1 { font-size: 20px; font-weight: 700; color: var(--navy); letter-spacing: -0.3px; }
        .header h1 span { color: var(--pink); }
        .nav { display: flex; gap: 4px; }
        .nav a {
            color: var(--gray-500); text-decoration: none; padding: 8px 16px;
            border-radius: 8px; font-size: 13px; font-weight: 500;
            transition: all 0.15s ease;
        }
        .nav a:hover { color: var(--navy); background: var(--gray-100); }
        .nav a.active { color: var(--pink); background: var(--pink-bg); font-weight: 600; }

        /* ── Layout ── */

        /* ── Sidebar ── */
        .sidebar {
            background: var(--white); padding: 24px 20px;
            border-right: 1px solid var(--gray-200); overflow-y: auto;
        }
        .sidebar h3 {
            font-size: 11px; text-transform: uppercase; color: var(--gray-500);
            margin-bottom: 12px; letter-spacing: 1.2px; font-weight: 600;
        }
        .agent-card {
            background: var(--gray-50); border-radius: var(--radius);
            padding: 14px 16px; margin-bottom: 8px; cursor: pointer;
            transition: all 0.15s ease; border: 1px solid var(--gray-200);
        }
        .agent-card:hover { border-color: var(--pink); box-shadow: var(--shadow-sm); }
        .agent-card .name { font-weight: 600; font-size: 13px; color: var(--navy); }
        .agent-card .status { font-size: 11px; color: var(--green); margin-top: 3px; font-weight: 500; }
        .tool-card .status { color: var(--gray-500); }

        /* ── Chat ── */
        .chat { display: flex; flex-direction: column; background: var(--white); }
        .chat-header {
            padding: 16px 24px; border-bottom: 1px solid var(--gray-200);
            display: flex; align-items: center; gap: 10px;
        }
        .chat-header .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
        .chat-header span { font-size: 14px; font-weight: 600; color: var(--navy); }
        .messages { flex: 1; overflow-y: auto; padding: 24px; background: var(--gray-50); }
        .message {
            max-width: 75%; margin-bottom: 14px; padding: 12px 16px;
            border-radius: 14px; line-height: 1.6; font-size: 14px;
            box-shadow: var(--shadow-sm);
        }
        .message.user {
            background: var(--navy); color: var(--white);
            margin-left: auto; border-bottom-right-radius: 4px;
        }
        .message.assistant {
            background: var(--white); color: var(--black);
            border: 1px solid var(--gray-200); border-bottom-left-radius: 4px;
        }
        .message.tool {
            background: var(--green-bg); color: var(--gray-900);
            border-left: 3px solid var(--green); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
        }
        .input-area { padding: 16px 24px; border-top: 1px solid var(--gray-200); background: var(--white); }
        .input-area form { display: flex; gap: 10px; }
        .input-area input {
            flex: 1; background: var(--gray-50); border: 1px solid var(--gray-200);
            border-radius: var(--radius); padding: 12px 16px; color: var(--black);
            font-size: 14px; font-family: inherit; outline: none; transition: border 0.15s;
        }
        .input-area input::placeholder { color: var(--gray-300); }
        .input-area input:focus { border-color: var(--pink); box-shadow: 0 0 0 3px rgba(214,51,108,0.1); }
        .input-area button {
            background: var(--pink); border: none; padding: 12px 28px;
            border-radius: var(--radius); color: var(--white); font-weight: 600;
            font-size: 14px; font-family: inherit; cursor: pointer;
            transition: background 0.15s;
        }
        .input-area button:hover { background: var(--pink-light); }

        /* ── Right Panel ── */
        .panel {
            background: var(--white); padding: 24px 20px;
            border-left: 1px solid var(--gray-200); overflow-y: auto;
        }
        .panel h3 {
            font-size: 11px; text-transform: uppercase; color: var(--gray-500);
            margin-bottom: 14px; letter-spacing: 1.2px; font-weight: 600;
        }
        .metric {
            background: var(--gray-50); border-radius: var(--radius);
            padding: 16px; margin-bottom: 10px; border: 1px solid var(--gray-200);
        }
        .metric .label { font-size: 11px; color: var(--gray-500); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric .value { font-size: 24px; font-weight: 700; margin-top: 6px; }
        .metric .value.navy { color: var(--navy); }
        .metric .value.pink { color: var(--pink); }
        .metric .value.green { color: var(--green); }
        .trace-item {
            background: var(--gray-50); border-radius: 8px; padding: 10px 14px;
            margin-bottom: 6px; font-size: 12px; border: 1px solid var(--gray-200);
            font-family: 'SF Mono', 'Fira Code', monospace; display: flex; justify-content: space-between;
        }
        .trace-item .span-name { color: var(--navy); font-weight: 600; }
        .trace-item .duration { color: var(--pink); font-weight: 500; }

        /* ── Empty State ── */
        .empty-state { text-align: center; padding: 80px 20px; }
        .empty-state .icon {
            width: 64px; height: 64px; border-radius: 16px; margin: 0 auto 20px;
            background: var(--pink-bg); display: flex; align-items: center; justify-content: center;
        }
        .empty-state .icon svg { width: 28px; height: 28px; }
        .empty-state h2 { font-size: 18px; color: var(--navy); font-weight: 700; margin-bottom: 8px; }
        .empty-state p { font-size: 14px; color: var(--gray-500); line-height: 1.6; }

        /* ── Tab Views ── */
        .view { display: none; height: calc(100vh - 64px); overflow-y: auto; background: var(--gray-50); }
        .view.active { display: block; }
        .view-chat.active { display: grid; grid-template-columns: 260px 1fr 300px; }
        .view-full { padding: 32px 48px; }
        .view-full h2 { font-size: 22px; font-weight: 700; color: var(--navy); margin-bottom: 6px; }
        .view-full .subtitle { font-size: 14px; color: var(--gray-500); margin-bottom: 28px; }

        /* ── Card Grid ── */
        .card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
        .card {
            background: var(--white); border: 1px solid var(--gray-200); border-radius: var(--radius);
            padding: 20px 24px; transition: all 0.15s ease;
        }
        .card:hover { border-color: var(--pink); box-shadow: var(--shadow-md); }
        .card .card-title { font-size: 15px; font-weight: 700; color: var(--navy); margin-bottom: 4px; }
        .card .card-category {
            display: inline-block; font-size: 11px; font-weight: 600; text-transform: uppercase;
            color: var(--pink); background: var(--pink-bg); padding: 2px 8px; border-radius: 4px;
            margin-bottom: 8px; letter-spacing: 0.5px;
        }
        .card .card-desc { font-size: 13px; color: var(--gray-700); line-height: 1.5; margin-bottom: 12px; }
        .card .card-tools { font-size: 11px; color: var(--gray-500); }
        .card .card-tools span {
            display: inline-block; background: var(--gray-100); padding: 2px 8px;
            border-radius: 4px; margin: 2px 4px 2px 0; font-family: 'SF Mono', 'Fira Code', monospace;
        }

        /* ── Data Table ── */
        .data-table { width: 100%; border-collapse: collapse; background: var(--white); border-radius: var(--radius); overflow: hidden; border: 1px solid var(--gray-200); }
        .data-table th {
            text-align: left; padding: 12px 16px; font-size: 11px; text-transform: uppercase;
            color: var(--gray-500); background: var(--gray-50); font-weight: 600; letter-spacing: 0.5px;
            border-bottom: 1px solid var(--gray-200);
        }
        .data-table td { padding: 12px 16px; font-size: 13px; color: var(--gray-700); border-bottom: 1px solid var(--gray-100); }
        .data-table tr:last-child td { border-bottom: none; }
        .data-table tr:hover td { background: var(--gray-50); }

        /* ── Info Blocks ── */
        .info-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin-bottom: 32px; }
        .info-block {
            background: var(--white); border: 1px solid var(--gray-200); border-radius: var(--radius);
            padding: 20px 24px;
        }
        .info-block h4 { font-size: 13px; font-weight: 700; color: var(--navy); margin-bottom: 10px; }
        .info-block ul { list-style: none; }
        .info-block li { font-size: 13px; color: var(--gray-700); padding: 4px 0; }
        .info-block li::before { content: "\\2022"; color: var(--pink); font-weight: 700; margin-right: 8px; }
        .badge {
            display: inline-block; font-size: 11px; font-weight: 600; padding: 3px 10px;
            border-radius: 20px; margin: 2px 4px 2px 0;
        }
        .badge-green { background: var(--green-bg); color: var(--green); }
        .badge-blue { background: var(--blue-bg); color: var(--blue); }
        .badge-pink { background: var(--pink-bg); color: var(--pink); }
        .badge-navy { background: var(--navy); color: var(--white); }
        .section-title { font-size: 15px; font-weight: 700; color: var(--navy); margin: 28px 0 14px; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--gray-300); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--gray-500); }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-brand">
            <div class="logo">B</div>
            <h1>Duxx <span>AI</span> Studio</h1>
        </div>
        <div class="nav">
            <a href="#" class="active">Chat</a>
            <a href="#agents">Agents</a>
            <a href="#traces">Traces</a>
            <a href="#finetune">Fine-tune</a>
            <a href="#governance">Governance</a>
            <a href="#cloud">Cloud</a>
        </div>
    </div>
    <!-- ── Chat View ── -->
    <div id="view-chat" class="view view-chat active">
        <div class="sidebar">
            <h3>Agents</h3>
            <div id="agents-list">
                <div class="agent-card">
                    <div class="name">Default Agent</div>
                    <div class="status">&#9679; Online</div>
                </div>
            </div>
            <h3 style="margin-top: 24px;">Tools</h3>
            <div id="tools-list"></div>
        </div>
        <div class="chat">
            <div class="chat-header">
                <div class="dot"></div>
                <span>Default Agent</span>
            </div>
            <div class="messages" id="messages">
                <div class="empty-state">
                    <div class="icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="#d6336c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                        </svg>
                    </div>
                    <h2>Welcome to Duxx AI Studio</h2>
                    <p>Start a conversation with your agent,<br>or configure agents in the sidebar.</p>
                </div>
            </div>
            <div class="input-area">
                <form id="chat-form">
                    <input type="text" id="chat-input" placeholder="Message your agent..." autocomplete="off" />
                    <button type="submit">Send</button>
                </form>
            </div>
        </div>
        <div class="panel">
            <h3>Metrics</h3>
            <div class="metric"><div class="label">Total Tokens</div><div class="value navy" id="metric-tokens">0</div></div>
            <div class="metric"><div class="label">Latency (avg)</div><div class="value green" id="metric-latency">0ms</div></div>
            <div class="metric"><div class="label">Cost</div><div class="value pink" id="metric-cost">$0.00</div></div>
            <h3 style="margin-top: 24px;">Trace</h3>
            <div id="trace-panel"></div>
        </div>
    </div>

    <!-- ── Agents View ── -->
    <div id="view-agents" class="view view-full">
        <h2>Agent Templates</h2>
        <p class="subtitle">14 enterprise-ready agent templates across 6 business domains</p>
        <div class="card-grid" id="templates-grid">
            <div class="empty-state"><p>Loading templates...</p></div>
        </div>
    </div>

    <!-- ── Traces View ── -->
    <div id="view-traces" class="view view-full">
        <h2>Traces</h2>
        <p class="subtitle">Execution traces from agent chat sessions</p>
        <table class="data-table" id="traces-table">
            <thead><tr><th>Span</th><th>Duration</th><th>Status</th></tr></thead>
            <tbody id="traces-body">
                <tr><td colspan="3" style="text-align:center;color:var(--gray-500);padding:40px;">No traces yet. Start a chat conversation to generate traces.</td></tr>
            </tbody>
        </table>
    </div>

    <!-- ── Fine-tune View ── -->
    <div id="view-finetune" class="view view-full">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;">
            <div>
                <h2 style="margin-bottom:4px;">Fine-tuning Pipeline</h2>
                <p class="subtitle" style="margin-bottom:0;">Train and fine-tune models using Unsloth / HuggingFace PEFT</p>
            </div>
            <button id="ft-new-job-btn" style="background:var(--pink);color:#fff;border:none;padding:10px 24px;border-radius:8px;font-weight:600;font-size:13px;cursor:pointer;font-family:inherit;">+ New Training Job</button>
        </div>

        <!-- Tab bar -->
        <div style="display:flex;gap:4px;background:var(--gray-100);padding:3px;border-radius:8px;width:fit-content;margin-bottom:24px;">
            <button class="ft-tab active" data-tab="ft-configure" style="padding:8px 20px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;border:none;font-family:inherit;background:var(--white);color:var(--navy);box-shadow:0 1px 3px rgba(0,0,0,0.08);">Configure</button>
            <button class="ft-tab" data-tab="ft-models" style="padding:8px 20px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;border:none;font-family:inherit;background:transparent;color:var(--gray-500);">Models</button>
            <button class="ft-tab" data-tab="ft-jobs" style="padding:8px 20px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;border:none;font-family:inherit;background:transparent;color:var(--gray-500);">Jobs</button>
            <button class="ft-tab" data-tab="ft-datasets" style="padding:8px 20px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;border:none;font-family:inherit;background:transparent;color:var(--gray-500);">Datasets</button>
            <button class="ft-tab" data-tab="ft-export" style="padding:8px 20px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;border:none;font-family:inherit;background:transparent;color:var(--gray-500);">Export</button>
        </div>

        <!-- Configure Tab -->
        <div id="ft-configure" class="ft-panel">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
                <!-- Left: Form -->
                <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;">
                    <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:20px;">Training Configuration</h3>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Base Model</label>
                            <select id="ft-model" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                                <option value="qwen2.5-7b">Qwen 2.5 7B</option>
                                <option value="qwen2.5-14b">Qwen 2.5 14B</option>
                                <option value="qwen2.5-coder-7b">Qwen 2.5 Coder 7B</option>
                                <option value="llama3-8b">LLaMA 3.1 8B</option>
                                <option value="mistral-7b">Mistral 7B</option>
                                <option value="phi3.5-mini">Phi-3.5 Mini</option>
                                <option value="phi4-14b">Phi-4 14B</option>
                                <option value="gemma2-9b">Gemma 2 9B</option>
                                <option value="deepseek-r1-8b">DeepSeek R1 8B</option>
                            </select>
                        </div>
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Method</label>
                            <select id="ft-method" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                                <option value="qlora" selected>QLoRA (4-bit)</option>
                                <option value="lora">LoRA (16-bit)</option>
                                <option value="full">Full Fine-tuning</option>
                            </select>
                        </div>
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Epochs</label>
                            <input id="ft-epochs" type="number" value="3" min="1" max="50" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                        </div>
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Learning Rate</label>
                            <select id="ft-lr" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                                <option value="2e-4" selected>2e-4 (Recommended)</option>
                                <option value="1e-4">1e-4</option>
                                <option value="5e-5">5e-5</option>
                                <option value="2e-5">2e-5</option>
                            </select>
                        </div>
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Batch Size</label>
                            <input id="ft-batch" type="number" value="2" min="1" max="32" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                        </div>
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Max Seq Length</label>
                            <select id="ft-seqlen" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                                <option value="1024">1024</option>
                                <option value="2048" selected>2048</option>
                                <option value="4096">4096</option>
                                <option value="8192">8192</option>
                            </select>
                        </div>
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">LoRA Rank (r)</label>
                            <select id="ft-lora-r" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                                <option value="8">8</option>
                                <option value="16" selected>16</option>
                                <option value="32">32</option>
                                <option value="64">64</option>
                            </select>
                        </div>
                        <div>
                            <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Chat Template</label>
                            <select id="ft-template" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                                <option value="chatml" selected>ChatML</option>
                                <option value="alpaca">Alpaca</option>
                                <option value="llama3">Llama 3</option>
                                <option value="phi3">Phi-3</option>
                            </select>
                        </div>
                    </div>
                    <div style="margin-top:20px;">
                        <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:6px;">Dataset Source</label>
                        <select id="ft-dataset" style="width:100%;padding:10px 12px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                            <option value="traces">Agent Traces (auto-collected)</option>
                            <option value="upload">Upload JSONL File</option>
                            <option value="huggingface">HuggingFace Dataset</option>
                        </select>
                    </div>
                    <div style="margin-top:24px;display:flex;gap:12px;">
                        <button id="ft-start-btn" style="background:var(--pink);color:#fff;border:none;padding:12px 28px;border-radius:8px;font-weight:600;font-size:14px;cursor:pointer;font-family:inherit;">Start Training</button>
                        <button style="background:var(--gray-100);color:var(--gray-700);border:1px solid var(--gray-200);padding:12px 28px;border-radius:8px;font-weight:600;font-size:14px;cursor:pointer;font-family:inherit;">Save Config</button>
                    </div>
                </div>
                <!-- Right: Live Status / Info -->
                <div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;margin-bottom:16px;">
                        <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:16px;">Pipeline Steps</h3>
                        <div id="ft-steps">
                            <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--gray-100);"><span style="width:24px;height:24px;border-radius:50%;background:var(--gray-100);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--gray-500);">1</span><span style="font-size:13px;color:var(--navy);font-weight:500;">Prepare Dataset</span><span style="margin-left:auto;font-size:11px;color:var(--gray-400);">Pending</span></div>
                            <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--gray-100);"><span style="width:24px;height:24px;border-radius:50%;background:var(--gray-100);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--gray-500);">2</span><span style="font-size:13px;color:var(--navy);font-weight:500;">Load Model + LoRA</span><span style="margin-left:auto;font-size:11px;color:var(--gray-400);">Pending</span></div>
                            <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--gray-100);"><span style="width:24px;height:24px;border-radius:50%;background:var(--gray-100);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--gray-500);">3</span><span style="font-size:13px;color:var(--navy);font-weight:500;">Train with SFTTrainer</span><span style="margin-left:auto;font-size:11px;color:var(--gray-400);">Pending</span></div>
                            <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--gray-100);"><span style="width:24px;height:24px;border-radius:50%;background:var(--gray-100);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--gray-500);">4</span><span style="font-size:13px;color:var(--navy);font-weight:500;">Evaluate</span><span style="margin-left:auto;font-size:11px;color:var(--gray-400);">Pending</span></div>
                            <div style="display:flex;align-items:center;gap:12px;padding:10px 0;"><span style="width:24px;height:24px;border-radius:50%;background:var(--gray-100);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--gray-500);">5</span><span style="font-size:13px;color:var(--navy);font-weight:500;">Export / GGUF</span><span style="margin-left:auto;font-size:11px;color:var(--gray-400);">Pending</span></div>
                        </div>
                    </div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;">
                        <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:16px;">Target Modules</h3>
                        <div style="display:flex;flex-wrap:wrap;gap:6px;">
                            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--pink-bg);color:var(--pink);padding:4px 10px;border-radius:6px;">q_proj</span>
                            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--pink-bg);color:var(--pink);padding:4px 10px;border-radius:6px;">k_proj</span>
                            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--pink-bg);color:var(--pink);padding:4px 10px;border-radius:6px;">v_proj</span>
                            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--pink-bg);color:var(--pink);padding:4px 10px;border-radius:6px;">o_proj</span>
                            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--blue-bg);color:var(--blue);padding:4px 10px;border-radius:6px;">gate_proj</span>
                            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--blue-bg);color:var(--blue);padding:4px 10px;border-radius:6px;">up_proj</span>
                            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--blue-bg);color:var(--blue);padding:4px 10px;border-radius:6px;">down_proj</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Models Tab -->
        <div id="ft-models" class="ft-panel" style="display:none;">
            <div class="card-grid" id="ft-models-grid"><p style="color:var(--gray-500);">Loading models...</p></div>
        </div>

        <!-- Jobs Tab -->
        <div id="ft-jobs" class="ft-panel" style="display:none;">
            <table class="data-table" id="ft-jobs-table">
                <thead><tr><th>ID</th><th>Name</th><th>Model</th><th>Method</th><th>Status</th><th>Progress</th><th>Loss</th></tr></thead>
                <tbody id="ft-jobs-body">
                    <tr><td colspan="7" style="text-align:center;color:var(--gray-500);padding:40px;">No training jobs yet. Click "New Training Job" to get started.</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Datasets Tab -->
        <div id="ft-datasets" class="ft-panel" style="display:none;">
            <!-- HuggingFace Popular Datasets -->
            <div style="background:linear-gradient(135deg,#fef3c7,#fde68a);border:1px solid #f59e0b;border-radius:10px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;">
                <span style="font-size:28px;">&#129303;</span>
                <div style="flex:1;">
                    <div style="font-size:14px;font-weight:700;color:#92400e;">HuggingFace Datasets Hub</div>
                    <div style="font-size:12px;color:#a16207;">Browse 26+ popular fine-tuning datasets. Click to load directly.</div>
                </div>
                <select id="ds-hf-category" style="padding:8px 12px;border:1px solid #f59e0b;border-radius:6px;font-size:12px;font-family:inherit;background:#fff;color:#92400e;">
                    <option value="">All Categories</option>
                </select>
                <button id="ds-browse-hf" style="background:#f59e0b;color:#fff;border:none;padding:8px 18px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;font-family:inherit;">Browse</button>
            </div>
            <div id="ds-hf-grid" style="display:none;margin-bottom:20px;max-height:300px;overflow-y:auto;display:none;"></div>

            <!-- Data Source Connectors -->
            <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:20px 24px;margin-bottom:20px;">
                <h3 style="font-size:14px;font-weight:700;color:var(--navy);margin-bottom:14px;">&#128268; Connect Data Source</h3>
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">
                    <button class="ds-src-btn" data-src="huggingface" style="padding:8px 14px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;background:var(--gray-50);color:var(--navy);">&#129303; HuggingFace</button>
                    <button class="ds-src-btn" data-src="local" style="padding:8px 14px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;background:var(--gray-50);color:var(--navy);">&#128193; Local File</button>
                    <button class="ds-src-btn" data-src="s3" style="padding:8px 14px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;background:var(--gray-50);color:var(--navy);">&#9729; AWS S3</button>
                    <button class="ds-src-btn" data-src="gcs" style="padding:8px 14px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;background:var(--gray-50);color:var(--navy);">&#9729; Google Cloud</button>
                    <button class="ds-src-btn" data-src="azure" style="padding:8px 14px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;background:var(--gray-50);color:var(--navy);">&#9729; Azure Blob</button>
                    <button class="ds-src-btn" data-src="database" style="padding:8px 14px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;background:var(--gray-50);color:var(--navy);">&#128451; Database</button>
                    <button class="ds-src-btn" data-src="google_drive" style="padding:8px 14px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;background:var(--gray-50);color:var(--navy);">&#128194; Google Drive</button>
                </div>
                <div id="ds-src-config" style="display:none;padding:14px;background:var(--gray-50);border-radius:8px;border:1px solid var(--gray-200);"></div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <!-- Left: Create + Split -->
                <div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:24px;margin-bottom:16px;">
                        <h3 style="font-size:14px;font-weight:700;color:var(--navy);margin-bottom:14px;">&#128221; Create / Paste Dataset</h3>
                        <div style="margin-bottom:12px;">
                            <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">Dataset Name</label>
                            <input id="ds-name" type="text" value="my-dataset" style="width:100%;padding:8px 10px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;font-family:inherit;background:var(--gray-50);color:var(--navy);">
                        </div>
                        <div style="margin-bottom:12px;">
                            <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">JSONL Data <span style="font-weight:400;color:var(--gray-400);">(supports CSV, JSON, JSONL — auto-converted)</span></label>
                            <textarea id="ds-data" rows="6" placeholder='{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}' style="width:100%;padding:8px 10px;border:1px solid var(--gray-200);border-radius:6px;font-size:11px;font-family:'JetBrains Mono',monospace;background:var(--gray-50);color:var(--navy);resize:vertical;"></textarea>
                        </div>
                        <div style="display:flex;gap:8px;">
                            <button id="ds-create-btn" style="background:var(--pink);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;font-family:inherit;">Create</button>
                            <button id="ds-sample-btn" style="background:var(--gray-100);color:var(--gray-700);border:1px solid var(--gray-200);padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;font-family:inherit;">Load Sample</button>
                        </div>
                    </div>
                    <!-- Split -->
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:24px;">
                        <h3 style="font-size:14px;font-weight:700;color:var(--navy);margin-bottom:14px;">&#9986; Train / Validation / Test Split</h3>
                        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:14px;">
                            <div><label style="font-size:10px;font-weight:700;color:var(--green);">TRAIN %</label><input id="ds-train" type="number" value="80" min="0" max="100" style="width:100%;padding:6px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;font-family:inherit;"></div>
                            <div><label style="font-size:10px;font-weight:700;color:var(--blue);">VALIDATION %</label><input id="ds-val" type="number" value="10" min="0" max="100" style="width:100%;padding:6px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;font-family:inherit;"></div>
                            <div><label style="font-size:10px;font-weight:700;color:var(--pink);">TEST %</label><input id="ds-test" type="number" value="10" min="0" max="100" style="width:100%;padding:6px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;font-family:inherit;"></div>
                        </div>
                        <button id="ds-split-btn" style="background:var(--navy);color:#fff;border:none;padding:8px 18px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;font-family:inherit;" disabled>Split Dataset</button>
                        <div id="ds-split-result" style="margin-top:10px;font-size:11px;color:var(--gray-500);"></div>
                    </div>
                </div>
                <!-- Right: Datasets List + Preview + Stats -->
                <div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:24px;margin-bottom:16px;">
                        <h3 style="font-size:14px;font-weight:700;color:var(--navy);margin-bottom:14px;">&#128202; My Datasets</h3>
                        <div id="ds-list" style="font-size:12px;color:var(--gray-500);max-height:120px;overflow-y:auto;">No datasets loaded yet.</div>
                    </div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:24px;margin-bottom:16px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                            <h3 style="font-size:14px;font-weight:700;color:var(--navy);margin:0;">Preview</h3>
                            <div style="display:flex;gap:6px;">
                                <button id="ds-validate-btn" style="background:var(--green-bg);color:var(--green);border:none;padding:5px 12px;border-radius:5px;font-weight:600;font-size:10px;cursor:pointer;font-family:inherit;" disabled>Validate</button>
                                <button id="ds-stats-btn" style="background:var(--blue-bg);color:var(--blue);border:none;padding:5px 12px;border-radius:5px;font-weight:600;font-size:10px;cursor:pointer;font-family:inherit;" disabled>Stats</button>
                            </div>
                        </div>
                        <div id="ds-preview" style="max-height:180px;overflow-y:auto;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--gray-600);">No data to preview.</div>
                    </div>
                    <div id="ds-stats-panel" style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:24px;display:none;">
                        <h3 style="font-size:14px;font-weight:700;color:var(--navy);margin-bottom:14px;">Statistics</h3>
                        <div id="ds-stats-content"></div>
                    </div>
                </div>
            </div>

            <!-- Model Export to Cloud -->
            <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:20px 24px;margin-top:20px;">
                <h3 style="font-size:14px;font-weight:700;color:var(--navy);margin-bottom:14px;">&#128640; Export Trained Model to Cloud</h3>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <div style="flex:1;min-width:200px;padding:14px;border:1px solid var(--gray-200);border-radius:8px;cursor:pointer;text-align:center;" onclick="alert('Configure AWS S3 bucket, region, and credentials to upload your trained model.')">
                        <div style="font-size:20px;margin-bottom:6px;">&#9729;</div>
                        <div style="font-size:12px;font-weight:600;color:var(--navy);">AWS S3</div>
                        <div style="font-size:10px;color:var(--gray-500);">s3://bucket/model</div>
                    </div>
                    <div style="flex:1;min-width:200px;padding:14px;border:1px solid var(--gray-200);border-radius:8px;cursor:pointer;text-align:center;" onclick="alert('Configure GCS bucket and credentials to upload your trained model.')">
                        <div style="font-size:20px;margin-bottom:6px;">&#9729;</div>
                        <div style="font-size:12px;font-weight:600;color:var(--navy);">Google Cloud</div>
                        <div style="font-size:10px;color:var(--gray-500);">gs://bucket/model</div>
                    </div>
                    <div style="flex:1;min-width:200px;padding:14px;border:1px solid var(--gray-200);border-radius:8px;cursor:pointer;text-align:center;" onclick="alert('Configure Azure Blob container and connection string to upload your trained model.')">
                        <div style="font-size:20px;margin-bottom:6px;">&#9729;</div>
                        <div style="font-size:12px;font-weight:600;color:var(--navy);">Azure Blob</div>
                        <div style="font-size:10px;color:var(--gray-500);">azure://container/model</div>
                    </div>
                    <div style="flex:1;min-width:200px;padding:14px;border:1px solid var(--gray-200);border-radius:8px;cursor:pointer;text-align:center;" onclick="alert('Push model to HuggingFace Hub. Requires HF_TOKEN.')">
                        <div style="font-size:20px;margin-bottom:6px;">&#129303;</div>
                        <div style="font-size:12px;font-weight:600;color:var(--navy);">HuggingFace Hub</div>
                        <div style="font-size:10px;color:var(--gray-500);">huggingface.co/model</div>
                    </div>
                    <div style="flex:1;min-width:200px;padding:14px;border:1px solid var(--pink);border-radius:8px;cursor:pointer;text-align:center;background:var(--pink-bg);" onclick="alert('Store on Duxx AI Cloud with auto-deployment, versioning, and team sharing. (Paid)')">
                        <div style="font-size:20px;margin-bottom:6px;">&#128171;</div>
                        <div style="font-size:12px;font-weight:700;color:var(--pink);">Duxx AI Cloud</div>
                        <div style="font-size:10px;color:var(--pink);">Managed &#8226; Auto-deploy &#8226; Paid</div>
                    </div>
                </div>
            </div>
        </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
                <!-- Left: Config -->
                <div>
                    <div style="background:linear-gradient(135deg,#0c1b3a,#1a2d5a);border-radius:10px;padding:24px;margin-bottom:16px;color:#fff;">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                            <span style="font-size:24px;">&#9729;</span>
                        </div>
                        <p style="font-size:12px;opacity:0.8;line-height:1.6;">Train on 40+ models (up to 397B params) with pay-per-token pricing. No local GPU required. LoRA adapters only.</p>
                    </div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;">
                        <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:20px;">Training Configuration</h3>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
                            <div style="grid-column:1/3;">
                                <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">Base Model</label>
                                <select id="tk-model" style="width:100%;padding:10px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);"></select>
                            </div>
                            <div>
                                <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">LoRA Rank</label>
                                <select id="tk-rank" style="width:100%;padding:10px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);">
                                    <option value="8">8</option><option value="16">16</option><option value="32" selected>32</option><option value="64">64</option><option value="128">128</option>
                                </select>
                            </div>
                            <div>
                                <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">Learning Rate</label>
                                <select id="tk-lr" style="width:100%;padding:10px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);">
                                    <option value="3e-4">3e-4</option><option value="1e-4" selected>1e-4</option><option value="5e-5">5e-5</option><option value="3e-5">3e-5</option><option value="1e-5">1e-5</option>
                                </select>
                            </div>
                            <div>
                                <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">Epochs</label>
                                <input id="tk-epochs" type="number" value="3" min="1" max="20" style="width:100%;padding:10px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);">
                            </div>
                            <div>
                                <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">Batch Size</label>
                                <input id="tk-batch" type="number" value="4" min="1" max="32" style="width:100%;padding:10px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);">
                            </div>
                            <div>
                                <label style="font-size:11px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px;">Loss Function</label>
                                <select id="tk-loss" style="width:100%;padding:10px;border:1px solid var(--gray-200);border-radius:8px;font-size:13px;font-family:inherit;background:var(--gray-50);">
                                    <option value="cross_entropy" selected>Cross Entropy (SL)</option>
                                    <option value="importance_sampling">Importance Sampling (RL)</option>
                                    <option value="ppo">PPO</option>
                                    <option value="dro">DRO</option>
                                </select>
                            </div>
                        </div>
                        <div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap;">
                            <label style="font-size:11px;display:flex;align-items:center;gap:4px;"><input type="checkbox" id="tk-mlp" checked> <span style="color:var(--gray-600);">Train MLP</span></label>
                            <label style="font-size:11px;display:flex;align-items:center;gap:4px;"><input type="checkbox" id="tk-attn" checked> <span style="color:var(--gray-600);">Train Attention</span></label>
                            <label style="font-size:11px;display:flex;align-items:center;gap:4px;"><input type="checkbox" id="tk-unembed" checked> <span style="color:var(--gray-600);">Train Unembed</span></label>
                        </div>
                        <div style="margin-top:20px;display:flex;gap:10px;">
                            <button id="tk-estimate-btn" style="background:var(--blue);color:#fff;border:none;padding:10px 20px;border-radius:8px;font-weight:600;font-size:13px;cursor:pointer;font-family:inherit;">Estimate Cost</button>
                            <button id="tk-train-btn" style="background:var(--pink);color:#fff;border:none;padding:10px 20px;border-radius:8px;font-weight:600;font-size:13px;cursor:pointer;font-family:inherit;">Start Training</button>
                        </div>
                        <div id="tk-cost-result" style="margin-top:12px;"></div>
                    </div>
                </div>
                <!-- Right: Models + Jobs -->
                <div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;margin-bottom:16px;">
                        <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:16px;">Pricing (per 1M tokens)</h3>
                        <div id="tk-pricing" style="max-height:300px;overflow-y:auto;"></div>
                    </div>
                    <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;">
                    </div>
                </div>
            </div>
        </div>

        <!-- Export Tab -->
        <div id="ft-export" class="ft-panel" style="display:none;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
                <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;">
                    <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:16px;">Export Formats</h3>
                    <div style="display:flex;flex-direction:column;gap:10px;">
                        <div style="padding:14px 16px;border:1px solid var(--gray-200);border-radius:8px;display:flex;align-items:center;gap:12px;cursor:pointer;"><span style="font-size:18px;">&#128190;</span><div><div style="font-size:13px;font-weight:600;color:var(--navy);">SafeTensors</div><div style="font-size:11px;color:var(--gray-500);">HuggingFace format, full precision</div></div></div>
                        <div style="padding:14px 16px;border:1px solid var(--gray-200);border-radius:8px;display:flex;align-items:center;gap:12px;cursor:pointer;"><span style="font-size:18px;">&#9881;</span><div><div style="font-size:13px;font-weight:600;color:var(--navy);">GGUF</div><div style="font-size:11px;color:var(--gray-500);">Quantized for llama.cpp, Ollama, vLLM</div></div></div>
                        <div style="padding:14px 16px;border:1px solid var(--gray-200);border-radius:8px;display:flex;align-items:center;gap:12px;cursor:pointer;"><span style="font-size:18px;">&#128200;</span><div><div style="font-size:13px;font-weight:600;color:var(--navy);">LoRA Adapter</div><div style="font-size:11px;color:var(--gray-500);">~100MB adapter only, merge at inference</div></div></div>
                        <div style="padding:14px 16px;border:1px solid var(--gray-200);border-radius:8px;display:flex;align-items:center;gap:12px;cursor:pointer;"><span style="font-size:18px;">&#128640;</span><div><div style="font-size:13px;font-weight:600;color:var(--navy);">Push to Hub</div><div style="font-size:11px;color:var(--gray-500);">Upload to HuggingFace Hub</div></div></div>
                    </div>
                </div>
                <div style="background:var(--white);border:1px solid var(--gray-200);border-radius:10px;padding:28px;">
                    <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:16px;">GGUF Quantization</h3>
                    <div id="ft-quant-list" style="display:flex;flex-direction:column;gap:6px;max-height:400px;overflow-y:auto;"><p style="color:var(--gray-500);">Loading...</p></div>
                </div>
            </div>
        </div>
    </div>

    <!-- ── Governance View ── -->
    <div id="view-governance" class="view view-full">
        <h2>Governance</h2>
        <p class="subtitle">Guardrails, role-based access control, and audit logging</p>

        <h3 class="section-title">Guardrail Types</h3>
        <div class="info-grid" id="guardrails-grid">
            <div class="info-block">
                <h4><span class="badge badge-pink">INPUT</span> PII Filter</h4>
                <ul><li>Detects and masks personally identifiable information</li><li>SSN, email, phone, credit card patterns</li></ul>
            </div>
            <div class="info-block">
                <h4><span class="badge badge-pink">INPUT</span> Prompt Injection</h4>
                <ul><li>Blocks prompt injection attempts</li><li>Pattern matching + heuristic detection</li></ul>
            </div>
            <div class="info-block">
                <h4><span class="badge badge-blue">OUTPUT</span> Content Filter</h4>
                <ul><li>Filters harmful or inappropriate content</li><li>Configurable severity thresholds</li></ul>
            </div>
            <div class="info-block">
                <h4><span class="badge badge-blue">OUTPUT</span> Hallucination Check</h4>
                <ul><li>Cross-references responses with source data</li><li>Flags unsupported claims</li></ul>
            </div>
            <div class="info-block">
                <h4><span class="badge badge-green">BUDGET</span> Token Budget</h4>
                <ul><li>Enforces maximum token spend per session</li><li>Configurable limits and alerts</li></ul>
            </div>
        </div>

        <h3 class="section-title">RBAC Roles</h3>
        <table class="data-table">
            <thead><tr><th>Role</th><th>Permissions</th><th>Description</th></tr></thead>
            <tbody>
                <tr><td><span class="badge badge-navy">ADMIN</span></td><td>All permissions</td><td>Full system access, user management, configuration</td></tr>
                <tr><td><span class="badge badge-pink">DEVELOPER</span></td><td>Create, configure, deploy agents</td><td>Build and manage agents, tools, and workflows</td></tr>
                <tr><td><span class="badge badge-blue">OPERATOR</span></td><td>Run agents, view traces</td><td>Execute agents and monitor performance</td></tr>
                <tr><td><span class="badge badge-green">VIEWER</span></td><td>Read-only access</td><td>View dashboards, traces, and audit logs</td></tr>
            </tbody>
        </table>

        <h3 class="section-title">Audit Logging</h3>
        <div class="info-grid">
            <div class="info-block">
                <h4>Tracked Events</h4>
                <ul>
                    <li>Agent creation and configuration changes</li>
                    <li>Tool executions and approvals</li>
                    <li>Guardrail triggers and violations</li>
                    <li>User access and role changes</li>
                </ul>
            </div>
            <div class="info-block">
                <h4>Export Formats</h4>
                <ul>
                    <li>JSON structured logs</li>
                    <li>OpenTelemetry spans</li>
                    <li>CSV export for compliance</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- ── Cloud View ── -->
    <div id="view-cloud" class="view view-full">
        <h2>Cloud Providers</h2>
        <p class="subtitle">Connect your cloud accounts for model storage, training, and deployment</p>

        <div class="info-grid">
            <!-- AWS Card -->
            <div class="info-block" id="aws-card" style="border: 2px solid #f59e0b;">
                <h4 style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">&#9729;</span> Amazon Web Services
                    <span id="aws-status-badge" class="badge badge-pink" style="margin-left:auto;">Not Connected</span>
                </h4>
                <div id="aws-account-info" style="display:none;margin:12px 0;padding:12px;background:#f8fafc;border-radius:8px;">
                    <div style="font-size:11px;color:#64748b;">Account: <strong id="aws-account-id">—</strong></div>
                    <div style="font-size:11px;color:#64748b;">Region: <strong id="aws-region-val">—</strong></div>
                    <div style="font-size:11px;color:#64748b;">IAM: <strong id="aws-iam-arn">—</strong></div>
                    <div style="font-size:11px;color:#64748b;">Method: <strong id="aws-auth-method">—</strong></div>
                </div>
                <div id="aws-connect-form">
                    <div style="margin:12px 0;">
                        <label style="font-size:11px;font-weight:600;color:#64748b;">Auth Method</label>
                        <select id="aws-auth-select" style="width:100%;padding:6px 8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;margin-top:4px;">
                            <option value="access_keys">Access Key + Secret (Recommended)</option>
                            <option value="sso">AWS SSO (Browser Login)</option>
                        </select>
                        <div style="font-size:10px;color:#94a3b8;margin-top:4px;">AWS CLI Profile and Environment Variables require local AWS CLI setup.</div>
                    </div>
                    <div id="aws-keys-fields">
                        <div style="margin:8px 0;">
                            <label style="font-size:11px;font-weight:600;color:#64748b;">Access Key ID</label>
                            <input id="aws-access-key" type="text" placeholder="AKIA..." style="width:100%;padding:6px 8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;font-family:monospace;margin-top:4px;">
                        </div>
                        <div style="margin:8px 0;">
                            <label style="font-size:11px;font-weight:600;color:#64748b;">Secret Access Key</label>
                            <input id="aws-secret-key" type="password" placeholder="Secret..." style="width:100%;padding:6px 8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;font-family:monospace;margin-top:4px;">
                        </div>
                        <div style="margin:8px 0;">
                            <label style="font-size:11px;font-weight:600;color:#64748b;">Region</label>
                            <select id="aws-region-select" style="width:100%;padding:6px 8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;margin-top:4px;">
                                <option>us-east-1</option><option>us-east-2</option><option>us-west-1</option><option>us-west-2</option>
                                <option>eu-west-1</option><option>eu-west-2</option><option>eu-central-1</option>
                                <option>ap-south-1</option><option>ap-southeast-1</option><option>ap-northeast-1</option>
                            </select>
                        </div>
                    </div>
                    <div id="aws-sso-fields" style="display:none;">
                        <div style="margin:8px 0;">
                            <label style="font-size:11px;font-weight:600;color:#64748b;">SSO Start URL</label>
                            <input id="aws-sso-url" type="text" placeholder="https://myorg.awsapps.com/start" style="width:100%;padding:6px 8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;margin-top:4px;">
                        </div>
                        <div style="margin:8px 0;">
                            <label style="font-size:11px;font-weight:600;color:#64748b;">Account ID</label>
                            <input id="aws-sso-account" type="text" placeholder="123456789012" style="width:100%;padding:6px 8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;margin-top:4px;">
                        </div>
                    </div>
                    <button id="aws-connect-btn" onclick="connectAWS()" style="margin-top:12px;background:#f59e0b;color:#fff;border:none;padding:8px 20px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;">Connect AWS</button>
                </div>
                <div id="aws-connected-actions" style="display:none;margin-top:12px;">
                    <button onclick="provisionAWS()" style="background:#16a34a;color:#fff;border:none;padding:8px 16px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;margin-right:8px;">&#9889; Auto-Provision Resources</button>
                    <button onclick="listBuckets()" style="background:#3b82f6;color:#fff;border:none;padding:8px 16px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;margin-right:8px;">&#128451; List Buckets</button>
                    <button onclick="disconnectAWS()" style="background:#fee2e2;color:#ef4444;border:none;padding:8px 16px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;">Disconnect</button>
                </div>
                <div id="aws-provision-results" style="display:none;margin-top:12px;"></div>
                <div id="aws-buckets-list" style="display:none;margin-top:12px;"></div>
            </div>

            <!-- Azure Card (stub) -->
            <div class="info-block" style="opacity:0.6;">
                <h4 style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">&#9729;</span> Microsoft Azure
                    <span class="badge badge-blue" style="margin-left:auto;">Coming Soon</span>
                </h4>
                <ul>
                    <li>Azure Blob Storage for models</li>
                    <li>Azure ML for training</li>
                    <li>Azure OpenAI Service</li>
                </ul>
            </div>

            <!-- GCP Card (stub) -->
            <div class="info-block" style="opacity:0.6;">
                <h4 style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">&#9729;</span> Google Cloud Platform
                    <span class="badge badge-green" style="margin-left:auto;">Coming Soon</span>
                </h4>
                <ul>
                    <li>Google Cloud Storage for models</li>
                    <li>Vertex AI for training</li>
                    <li>Google AI Studio</li>
                </ul>
            </div>
        </div>
    </div>
    <script>
        /* ── AWS Cloud Functions ── */
        document.getElementById('aws-auth-select')?.addEventListener('change', function() {
            const v = this.value;
            document.getElementById('aws-keys-fields').style.display = (v === 'access_keys') ? 'block' : 'none';
            document.getElementById('aws-sso-fields').style.display = (v === 'sso') ? 'block' : 'none';
        });

        async function connectAWS() {
            const method = document.getElementById('aws-auth-select').value;
            const body = { auth_method: method, region: document.getElementById('aws-region-select')?.value || 'us-east-1' };
            if (method === 'access_keys') {
                body.access_key = document.getElementById('aws-access-key').value;
                body.secret_key = document.getElementById('aws-secret-key').value;
            } else if (method === 'sso') {
                body.start_url = document.getElementById('aws-sso-url').value;
                body.account_id = document.getElementById('aws-sso-account').value;
            }
            const btn = document.getElementById('aws-connect-btn');
            btn.textContent = 'Connecting...'; btn.disabled = true;
            try {
                // Validate inputs
                if (method === 'access_keys' && (!body.access_key || !body.secret_key)) {
                    alert('Please enter both Access Key ID and Secret Access Key.');
                    btn.textContent = 'Connect AWS'; btn.disabled = false;
                    return;
                }
                if (method === 'sso' && !body.start_url) {
                    alert('Please enter the AWS SSO Start URL.');
                    btn.textContent = 'Connect AWS'; btn.disabled = false;
                    return;
                }
                const url = method === 'sso' ? '/api/cloud/aws/connect-sso' : '/api/cloud/aws/connect';
                const resp = await fetch(url, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
                const data = await resp.json();
                if (data.connected) {
                    showAWSConnected(data);
                } else if (data.verification_uri) {
                    window.open(data.verification_uri, '_blank');
                    btn.textContent = 'Waiting for SSO...';
                    pollSSO();
                } else {
                    const errMsg = data.error || data.detail || JSON.stringify(data);
                    alert('Connection failed:\\n' + errMsg);
                    btn.textContent = 'Connect AWS'; btn.disabled = false;
                }
            } catch(e) { alert('Connection error: ' + (e.message || String(e))); btn.textContent = 'Connect AWS'; btn.disabled = false; }
        }

        async function pollSSO() {
            for (let i = 0; i < 60; i++) {
                await new Promise(r => setTimeout(r, 5000));
                const resp = await fetch('/api/cloud/aws/sso-callback', { method: 'POST' });
                const data = await resp.json();
                if (data.connected) { showAWSConnected(data); return; }
                if (data.error && data.error !== 'authorization_pending' && data.error !== 'slow_down') {
                    alert('SSO failed: ' + data.error);
                    document.getElementById('aws-connect-btn').textContent = 'Connect AWS';
                    document.getElementById('aws-connect-btn').disabled = false;
                    return;
                }
            }
        }

        function showAWSConnected(data) {
            document.getElementById('aws-status-badge').textContent = 'Connected';
            document.getElementById('aws-status-badge').className = 'badge badge-green';
            document.getElementById('aws-account-id').textContent = data.account_id || '—';
            document.getElementById('aws-region-val').textContent = data.region || '—';
            document.getElementById('aws-iam-arn').textContent = data.iam_arn || '—';
            document.getElementById('aws-auth-method').textContent = data.auth_method || '—';
            document.getElementById('aws-account-info').style.display = 'block';
            document.getElementById('aws-connect-form').style.display = 'none';
            document.getElementById('aws-connected-actions').style.display = 'block';
            document.getElementById('aws-card').style.borderColor = '#16a34a';
        }

        async function provisionAWS() {
            const resp = await fetch('/api/cloud/aws/provision', { method: 'POST' });
            const data = await resp.json();
            const el = document.getElementById('aws-provision-results');
            el.style.display = 'block';
            if (data.results) {
                el.innerHTML = data.results.map(r =>
                    '<div style="padding:8px;margin:4px 0;background:' + (r.action === 'error' ? '#fef2f2' : r.action === 'created' ? '#f0fdf4' : '#f8fafc') +
                    ';border-radius:6px;font-size:11px;border:1px solid ' + (r.action === 'error' ? '#fecaca' : r.action === 'created' ? '#bbf7d0' : '#e2e8f0') + ';">' +
                    '<strong>' + r.resource + '</strong>: ' + r.action + (r.details ? ' — ' + r.details : '') + (r.error ? ' — ' + r.error : '') + '</div>'
                ).join('');
            } else {
                el.innerHTML = '<div style="color:#ef4444;font-size:12px;">' + (data.error || 'Failed') + '</div>';
            }
        }

        async function listBuckets() {
            const resp = await fetch('/api/cloud/aws/buckets');
            const buckets = await resp.json();
            const el = document.getElementById('aws-buckets-list');
            el.style.display = 'block';
            if (buckets.length > 0) {
                el.innerHTML = '<div style="font-size:11px;font-weight:600;color:#64748b;margin-bottom:6px;">S3 Buckets (' + buckets.length + ')</div>' +
                    buckets.map(b => '<div style="padding:6px 10px;margin:3px 0;background:#f8fafc;border-radius:4px;font-size:11px;font-family:monospace;border:1px solid #e2e8f0;">' + b.name + '</div>').join('');
            } else {
                el.innerHTML = '<div style="font-size:12px;color:#94a3b8;">No S3 buckets found.</div>';
            }
        }

        async function disconnectAWS() {
            await fetch('/api/cloud/aws/disconnect', { method: 'POST' });
            document.getElementById('aws-status-badge').textContent = 'Not Connected';
            document.getElementById('aws-status-badge').className = 'badge badge-pink';
            document.getElementById('aws-account-info').style.display = 'none';
            document.getElementById('aws-connect-form').style.display = 'block';
            document.getElementById('aws-connected-actions').style.display = 'none';
            document.getElementById('aws-provision-results').style.display = 'none';
            document.getElementById('aws-buckets-list').style.display = 'none';
            document.getElementById('aws-card').style.borderColor = '#f59e0b';
            document.getElementById('aws-connect-btn').textContent = 'Connect AWS';
            document.getElementById('aws-connect-btn').disabled = false;
        }

        // Check AWS status on Cloud tab load
        if (window.location.hash === '#cloud') {
            fetch('/api/cloud/aws/status').then(r => r.json()).then(data => {
                if (data.connected) showAWSConnected(data);
            }).catch(() => {});
        }

        /* ── Hash Router ── */
        const views = { '': 'view-chat', 'agents': 'view-agents', 'traces': 'view-traces', 'finetune': 'view-finetune', 'governance': 'view-governance', 'cloud': 'view-cloud' };
        const loaded = {};

        function navigate() {
            const hash = window.location.hash.replace('#', '') || '';
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            const viewId = views[hash] || 'view-chat';
            const el = document.getElementById(viewId);
            if (el) el.classList.add('active');
            document.querySelectorAll('.nav a').forEach(a => {
                const aHash = a.getAttribute('href').replace('#', '');
                a.classList.toggle('active', aHash === hash || (aHash === '' && hash === ''));
            });
            if (hash === 'agents' && !loaded.agents) loadTemplates();
        }

        window.addEventListener('hashchange', navigate);
        document.addEventListener('DOMContentLoaded', navigate);

        /* ── Load Templates ── */
        function loadTemplates() {
            loaded.agents = true;
            fetch('/api/templates').then(r => r.json()).then(templates => {
                const grid = document.getElementById('templates-grid');
                grid.innerHTML = templates.map(t => {
                    const tools = (t.tools || []).map(n => '<span>' + n + '</span>').join('');
                    return '<div class="card">' +
                        '<div class="card-category">' + t.category + '</div>' +
                        '<div class="card-title">' + t.name + '</div>' +
                        '<div class="card-desc">' + t.description + '</div>' +
                        '<div class="card-tools">' + tools + '</div>' +
                        '</div>';
                }).join('');
            }).catch(() => {
                document.getElementById('templates-grid').innerHTML = '<div class="empty-state"><p>Failed to load templates.</p></div>';
            });
        }

        /* ── Chat ── */
        const messagesEl = document.getElementById('messages');
        const form = document.getElementById('chat-form');
        const input = document.getElementById('chat-input');
        let firstMessage = true;
        const allTraces = [];

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = input.value.trim();
            if (!text) return;

            if (firstMessage) { messagesEl.innerHTML = ''; firstMessage = false; }
            addMessage('user', text);
            input.value = '';

            try {
                const resp = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await resp.json();
                addMessage('assistant', data.response);
                updateMetrics(data.metrics || {});
                updateTraces(data.traces || []);
            } catch (err) {
                addMessage('assistant', 'Error: ' + err.message);
            }
        });

        function addMessage(role, content) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.textContent = content;
            messagesEl.appendChild(div);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function updateMetrics(m) {
            if (m.tokens !== undefined) document.getElementById('metric-tokens').textContent = m.tokens.toLocaleString();
            if (m.latency !== undefined) document.getElementById('metric-latency').textContent = m.latency + 'ms';
            if (m.cost !== undefined) document.getElementById('metric-cost').textContent = '$' + m.cost.toFixed(4);
        }

        function updateTraces(traces) {
            const panel = document.getElementById('trace-panel');
            panel.innerHTML = traces.map(t =>
                '<div class="trace-item"><span class="span-name">' + t.name + '</span><span class="duration">' + t.duration + 'ms</span></div>'
            ).join('');
            // Also update traces view
            traces.forEach(t => allTraces.push(t));
            const tbody = document.getElementById('traces-body');
            if (allTraces.length > 0) {
                tbody.innerHTML = allTraces.map(t =>
                    '<tr><td>' + t.name + '</td><td>' + t.duration + 'ms</td><td><span class="badge badge-green">OK</span></td></tr>'
                ).join('');
            }
        }

        /* ── Load Tools Sidebar ── */
        fetch('/api/tools').then(r => r.json()).then(tools => {
            const list = document.getElementById('tools-list');
            list.innerHTML = tools.map(t =>
                '<div class="agent-card tool-card"><div class="name">' + t.name + '</div><div class="status">' + t.description.substring(0, 45) + '</div></div>'
            ).join('');
        }).catch(() => {});

        /* ── Fine-tune Tab Logic ── */
        const ftLoaded = {};
        document.querySelectorAll('.ft-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.ft-tab').forEach(t => {
                    t.classList.remove('active');
                    t.style.background = 'transparent'; t.style.color = 'var(--gray-500)'; t.style.boxShadow = 'none';
                });
                tab.classList.add('active');
                tab.style.background = 'var(--white)'; tab.style.color = 'var(--navy)'; tab.style.boxShadow = '0 1px 3px rgba(0,0,0,0.08)';
                document.querySelectorAll('.ft-panel').forEach(p => p.style.display = 'none');
                const target = tab.getAttribute('data-tab');
                document.getElementById(target).style.display = 'block';
                if (target === 'ft-models' && !ftLoaded.models) loadFTModels();
                if (target === 'ft-jobs') loadFTJobs();
                if (target === 'ft-export' && !ftLoaded.quant) loadFTQuant();
                if (target === 'ft-datasets' && !ftLoaded.hfCategories) loadHFCategories();
            });
        });

        function loadFTModels() {
            ftLoaded.models = true;
            fetch('/api/finetune/models').then(r => r.json()).then(models => {
                const grid = document.getElementById('ft-models-grid');
                grid.innerHTML = models.map(m => {
                    const tierBadge = m.tier === 'cloud'
                        ? '<span style="font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;background:var(--pink-bg);color:var(--pink);text-transform:uppercase;margin-left:8px;">Cloud</span>'
                        : '<span style="font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;background:var(--green-bg);color:var(--green);text-transform:uppercase;margin-left:8px;">OSS</span>';
                    return '<div class="card"><div class="card-title">' + m.name + tierBadge + '</div>' +
                        '<div style="font-size:12px;color:var(--gray-500);margin:6px 0;">Family: ' + m.family + ' &bull; ' + m.params + ' params &bull; Context: ' + m.context_length.toLocaleString() + '</div>' +
                        '<div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">' +
                        '<span style="font-size:11px;background:var(--gray-100);padding:2px 8px;border-radius:4px;">r=' + m.recommended_lora_r + '</span>' +
                        '<span style="font-size:11px;background:var(--gray-100);padding:2px 8px;border-radius:4px;">alpha=' + m.recommended_lora_alpha + '</span>' +
                        '<span style="font-size:11px;background:var(--gray-100);padding:2px 8px;border-radius:4px;">' + m.chat_template + '</span>' +
                        '<span style="font-size:11px;background:var(--gray-100);padding:2px 8px;border-radius:4px;">' + m.license + '</span>' +
                        '</div></div>';
                }).join('');
            }).catch(() => {});
        }

        function loadFTJobs() {
            fetch('/api/finetune/jobs').then(r => r.json()).then(jobs => {
                const tbody = document.getElementById('ft-jobs-body');
                if (jobs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--gray-500);padding:40px;">No training jobs yet.</td></tr>';
                    return;
                }
                tbody.innerHTML = jobs.map(j => {
                    const statusColors = { completed: 'var(--green)', training: 'var(--blue)', failed: '#ef4444', pending: 'var(--gray-500)', cancelled: 'var(--gray-400)' };
                    const color = statusColors[j.status] || 'var(--gray-500)';
                    return '<tr><td style="font-family:monospace;font-size:12px;">' + j.id + '</td><td>' + j.name + '</td>' +
                        '<td style="font-size:12px;">' + j.base_model.split('/').pop() + '</td>' +
                        '<td>' + j.method + '</td>' +
                        '<td><span style="font-size:11px;font-weight:600;padding:3px 10px;border-radius:12px;background:' + color + '15;color:' + color + ';">' + j.status + '</span></td>' +
                        '<td>' + j.progress.toFixed(0) + '%</td>' +
                        '<td>' + (j.current_loss > 0 ? j.current_loss.toFixed(4) : '-') + '</td></tr>';
                }).join('');
            }).catch(() => {});
        }

        function loadFTQuant() {
            ftLoaded.quant = true;
            fetch('/api/finetune/quant-methods').then(r => r.json()).then(methods => {
                const list = document.getElementById('ft-quant-list');
                list.innerHTML = methods.map(m => {
                    const qualColors = { low: '#ef4444', medium: 'var(--orange)', high: 'var(--blue)', very_high: 'var(--green)' };
                    const color = qualColors[m.quality] || 'var(--gray-500)';
                    return '<div style="padding:10px 14px;border:1px solid var(--gray-100);border-radius:8px;display:flex;align-items:center;gap:12px;">' +
                        '<code style="font-size:12px;font-weight:600;color:var(--navy);min-width:60px;">' + m.name + '</code>' +
                        '<div style="flex:1;font-size:11px;color:var(--gray-500);">' + m.bits + '-bit &bull; ~' + (m.size_ratio * 100).toFixed(0) + '% of f16 size</div>' +
                        '<span style="font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;background:' + color + '15;color:' + color + ';text-transform:uppercase;">' + m.quality + '</span></div>';
                }).join('');
            }).catch(() => {});
        }

        // Start training button
        document.getElementById('ft-start-btn').addEventListener('click', () => {
            const req = {
                name: 'duxx_ai-ft-' + Date.now(),
                base_model: document.getElementById('ft-model').value,
                method: document.getElementById('ft-method').value,
                epochs: parseInt(document.getElementById('ft-epochs').value),
                learning_rate: parseFloat(document.getElementById('ft-lr').value),
                batch_size: parseInt(document.getElementById('ft-batch').value),
                max_seq_length: parseInt(document.getElementById('ft-seqlen').value),
                lora_r: parseInt(document.getElementById('ft-lora-r').value),
                chat_template: document.getElementById('ft-template').value,
                dataset_source: document.getElementById('ft-dataset').value,
            };
            fetch('/api/finetune/jobs', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(req) })
                .then(r => r.json())
                .then(job => {
                    alert('Training job created: ' + job.id + '\\nStatus: ' + job.status + '\\n\\nNote: Actual GPU training requires CUDA + Unsloth installed.\\nUse CLI: duxx_ai finetune train');
                    // Switch to Jobs tab
                    document.querySelector('[data-tab="ft-jobs"]').click();
                }).catch(err => alert('Error: ' + err.message));
        });

        document.getElementById('ft-new-job-btn').addEventListener('click', () => {
            document.querySelector('[data-tab="ft-configure"]').click();
        });

        /* ── HuggingFace + Connectors Logic ── */
        function loadHFCategories() {
            ftLoaded.hfCategories = true;
            fetch('/api/datasets/categories').then(r => r.json()).then(cats => {
                const sel = document.getElementById('ds-hf-category');
                cats.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; sel.appendChild(o); });
            });
        }

        document.getElementById('ds-browse-hf').addEventListener('click', () => {
            const cat = document.getElementById('ds-hf-category').value;
            const url = cat ? '/api/datasets/popular?category=' + encodeURIComponent(cat) : '/api/datasets/popular';
            fetch(url).then(r => r.json()).then(datasets => {
                const grid = document.getElementById('ds-hf-grid');
                grid.style.display = 'block';
                grid.innerHTML = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;">' +
                    datasets.map(d => {
                        const catColor = {'Instruction':'var(--blue)','Chat':'var(--green)','Code':'var(--purple)','Math':'var(--orange)','Preference':'var(--pink)','Evaluation':'var(--gray-500)'}[d.category] || 'var(--navy)';
                        return '<div style="background:var(--white);border:1px solid var(--gray-200);border-radius:8px;padding:14px;cursor:pointer;" onclick="loadHFDataset(\\'' + d.hf_id + '\\')">' +
                            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;"><span style="font-size:13px;font-weight:600;color:var(--navy);">' + d.name + '</span><span style="font-size:10px;padding:2px 6px;border-radius:4px;background:' + catColor + '15;color:' + catColor + ';font-weight:600;">' + d.category + '</span></div>' +
                            '<div style="font-size:11px;color:var(--gray-500);margin-bottom:6px;line-height:1.4;">' + d.description.substring(0, 80) + '...</div>' +
                            '<div style="display:flex;gap:8px;font-size:10px;color:var(--gray-400);"><span>' + d.samples + ' samples</span><span>' + d.format + '</span></div></div>';
                    }).join('') + '</div>';
            });
        });

        window.loadHFDataset = function(hfId) {
            const max = prompt('Max samples to load (default 1000):', '1000');
            if (max === null) return;
            document.getElementById('ds-hf-grid').innerHTML = '<div style="padding:20px;text-align:center;color:var(--gray-500);">Loading ' + hfId + '...</div>';
            fetch('/api/datasets/load-hf', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dataset_name: hfId, split: 'train', max_samples: parseInt(max) || 1000})
            }).then(r => r.json()).then(res => {
                if (res.error) { alert('Error: ' + res.error); return; }
                currentDatasetId = res.id;
                document.getElementById('ds-hf-grid').style.display = 'none';
                document.getElementById('ds-split-btn').disabled = false;
                document.getElementById('ds-validate-btn').disabled = false;
                document.getElementById('ds-stats-btn').disabled = false;
                loadDatasetList();
                loadDatasetPreview(res.id);
                alert('Loaded ' + res.samples + ' samples from ' + hfId + ' (format: ' + res.format + ')');
            }).catch(e => alert('Failed: ' + e.message));
        };

        // Data source connector buttons
        document.querySelectorAll('.ds-src-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const src = btn.getAttribute('data-src');
                const cfg = document.getElementById('ds-src-config');
                cfg.style.display = 'block';
                const configs = {
                    'huggingface': '<div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--navy);">HuggingFace Dataset</div><input id="src-hf-name" placeholder="e.g. tatsu-lab/alpaca" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:8px;"><button onclick="loadHFDataset(document.getElementById(\\'src-hf-name\\').value)" style="background:var(--pink);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Load</button>',
                    'local': '<div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--navy);">Local File Path</div><input id="src-local-path" placeholder="C:/data/train.jsonl (JSONL, CSV, Excel, Parquet, JSON, TXT, PDF)" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:8px;"><button onclick="alert(\\'Local file loading requires backend file access. Use CLI: duxx_ai finetune prepare <path>\\')" style="background:var(--navy);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Load File</button>',
                    's3': '<div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--navy);">AWS S3</div><input placeholder="Bucket" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:6px;"><input placeholder="Key (e.g. data/train.jsonl)" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:6px;"><input placeholder="Region (us-east-1)" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:8px;"><div style="font-size:10px;color:var(--gray-400);margin-bottom:8px;">Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars or IAM role</div><button style="background:var(--navy);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Connect &amp; Load</button>',
                    'gcs': '<div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--navy);">Google Cloud Storage</div><input placeholder="Bucket name" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:6px;"><input placeholder="Blob path (e.g. datasets/train.jsonl)" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:8px;"><div style="font-size:10px;color:var(--gray-400);margin-bottom:8px;">Requires GOOGLE_APPLICATION_CREDENTIALS env var</div><button style="background:var(--navy);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Connect &amp; Load</button>',
                    'azure': '<div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--navy);">Azure Blob Storage</div><input placeholder="Container name" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:6px;"><input placeholder="Blob name" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:8px;"><div style="font-size:10px;color:var(--gray-400);margin-bottom:8px;">Requires AZURE_STORAGE_CONNECTION_STRING env var</div><button style="background:var(--navy);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Connect &amp; Load</button>',
                    'database': '<div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--navy);">Database Connection</div><select style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:6px;"><option>PostgreSQL</option><option>MySQL</option><option>SQLite</option><option>MongoDB</option></select><input placeholder="Connection string" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:6px;"><input placeholder="SQL query or collection name" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:8px;"><button style="background:var(--navy);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Connect &amp; Load</button>',
                    'google_drive': '<div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--navy);">Google Drive</div><input placeholder="File ID (from URL)" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:6px;"><input placeholder="Service account credentials path" style="width:100%;padding:8px;border:1px solid var(--gray-200);border-radius:6px;font-size:12px;margin-bottom:8px;"><button style="background:var(--navy);color:#fff;border:none;padding:8px 16px;border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Connect &amp; Load</button>',
                };
                cfg.innerHTML = configs[src] || '<p>Unknown source type</p>';
            });
        });

        /* ── Dataset Logic ── */
        let currentDatasetId = null;

        document.getElementById('ds-sample-btn').addEventListener('click', () => {
            document.getElementById('ds-data').value = [
                '{"messages": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a branch of AI that enables systems to learn from data."}]}',
                '{"messages": [{"role": "user", "content": "Explain neural networks"}, {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks."}]}',
                '{"messages": [{"role": "user", "content": "What is fine-tuning?"}, {"role": "assistant", "content": "Fine-tuning adapts a pre-trained model to a specific task using domain data."}]}',
                '{"messages": [{"role": "user", "content": "What is LoRA?"}, {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to freeze model weights."}]}',
                '{"messages": [{"role": "user", "content": "What is RAG?"}, {"role": "assistant", "content": "RAG retrieves relevant documents and uses them to augment LLM generation."}]}'
            ].join('\\n');
        });

        document.getElementById('ds-create-btn').addEventListener('click', () => {
            const name = document.getElementById('ds-name').value || 'my-dataset';
            const raw = document.getElementById('ds-data').value.trim();
            const lines = raw.split('\\n').filter(l => l.trim());
            let samples;
            try { samples = lines.map(l => JSON.parse(l)); }
            catch(e) { alert('Invalid JSONL: ' + e.message); return; }

            fetch('/api/finetune/datasets', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name, samples})
            }).then(r => r.json()).then(res => {
                currentDatasetId = res.id;
                document.getElementById('ds-split-btn').disabled = false;
                document.getElementById('ds-validate-btn').disabled = false;
                document.getElementById('ds-stats-btn').disabled = false;
                loadDatasetList();
                loadDatasetPreview(res.id);
            });
        });

        function loadDatasetList() {
            fetch('/api/finetune/datasets').then(r => r.json()).then(datasets => {
                const el = document.getElementById('ds-list');
                if (datasets.length === 0) { el.innerHTML = 'No datasets loaded yet.'; return; }
                el.innerHTML = datasets.map(d =>
                    '<div style="padding:10px 14px;border:1px solid var(--gray-100);border-radius:8px;margin-bottom:6px;display:flex;align-items:center;gap:10px;cursor:pointer;" onclick="loadDatasetPreview(\\'' + d.id + '\\')">' +
                    '<span style="font-weight:600;font-size:13px;color:var(--navy);">' + d.name + '</span>' +
                    '<span style="font-size:11px;color:var(--gray-500);">' + d.total_samples + ' samples</span>' +
                    '<span style="font-size:10px;padding:2px 8px;border-radius:4px;background:var(--green-bg);color:var(--green);font-weight:600;margin-left:auto;">' + d.status + '</span>' +
                    '</div>'
                ).join('');
            });
        }

        window.loadDatasetPreview = function(dsId) {
            currentDatasetId = dsId;
            fetch('/api/finetune/datasets/' + dsId + '/preview?n=10').then(r => r.json()).then(res => {
                const el = document.getElementById('ds-preview');
                if (res.error) { el.innerHTML = res.error; return; }
                el.innerHTML = '<table style="width:100%;font-size:11px;"><thead><tr><th style="text-align:left;padding:4px 8px;color:var(--gray-500);">#</th><th style="text-align:left;padding:4px 8px;color:var(--gray-500);">Format</th><th style="text-align:left;padding:4px 8px;color:var(--gray-500);">Tokens</th><th style="text-align:left;padding:4px 8px;color:var(--gray-500);">Content</th></tr></thead><tbody>' +
                    res.samples.map(s => '<tr><td style="padding:4px 8px;">' + s.line + '</td><td style="padding:4px 8px;"><span style="font-size:10px;padding:1px 6px;border-radius:3px;background:var(--blue-bg);color:var(--blue);">' + s.format + '</span></td><td style="padding:4px 8px;">' + s.tokens + '</td><td style="padding:4px 8px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + JSON.stringify(s.data).substring(0, 80) + '...</td></tr>').join('') +
                    '</tbody></table>';
            });
        };

        document.getElementById('ds-validate-btn').addEventListener('click', () => {
            if (!currentDatasetId) return;
            fetch('/api/finetune/datasets/' + currentDatasetId + '/validate', {method: 'POST'}).then(r => r.json()).then(res => {
                const icon = res.valid ? '&#9989;' : '&#10060;';
                alert(icon + ' Validation: ' + res.valid_samples + '/' + res.total_samples + ' valid\\nFormat: ' + res.detected_format + (res.errors.length > 0 ? '\\nErrors: ' + res.errors.join(', ') : ''));
                loadDatasetList();
            });
        });

        document.getElementById('ds-stats-btn').addEventListener('click', () => {
            if (!currentDatasetId) return;
            fetch('/api/finetune/datasets/' + currentDatasetId + '/stats').then(r => r.json()).then(stats => {
                const panel = document.getElementById('ds-stats-panel');
                panel.style.display = 'block';
                document.getElementById('ds-stats-content').innerHTML =
                    '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px;">' +
                    '<div style="background:var(--gray-50);padding:12px;border-radius:8px;text-align:center;"><div style="font-size:22px;font-weight:800;color:var(--navy);">' + stats.total_samples + '</div><div style="font-size:10px;color:var(--gray-500);">Samples</div></div>' +
                    '<div style="background:var(--gray-50);padding:12px;border-radius:8px;text-align:center;"><div style="font-size:22px;font-weight:800;color:var(--pink);">' + stats.total_tokens.toLocaleString() + '</div><div style="font-size:10px;color:var(--gray-500);">Total Tokens</div></div>' +
                    '<div style="background:var(--gray-50);padding:12px;border-radius:8px;text-align:center;"><div style="font-size:22px;font-weight:800;color:var(--blue);">' + Math.round(stats.avg_tokens) + '</div><div style="font-size:10px;color:var(--gray-500);">Avg Tokens</div></div></div>' +
                    '<div style="font-size:12px;color:var(--gray-600);"><strong>Token Distribution:</strong><div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">' +
                    Object.entries(stats.token_distribution).map(([k,v]) => '<span style="font-size:10px;padding:3px 8px;border-radius:4px;background:var(--gray-100);">' + k + ': ' + v + '</span>').join('') +
                    '</div></div>';
            });
        });

                const sel = document.getElementById('tk-model');
                sel.innerHTML = models.map(m => {
                    const tag = m.architecture === 'moe' ? ' (MoE)' : '';
                    const vis = m.vision ? ' 👁' : '';
                    return '<option value="' + m.id + '">' + m.name + tag + vis + ' — $' + m.price_train + '/1M tok</option>';
                }).join('');

                const pricing = document.getElementById('tk-pricing');
                pricing.innerHTML = '<table style="width:100%;font-size:11px;"><thead><tr><th style="text-align:left;padding:4px 6px;color:var(--gray-500);">Model</th><th style="text-align:right;padding:4px 6px;color:var(--gray-500);">Train</th><th style="text-align:right;padding:4px 6px;color:var(--gray-500);">Sample</th></tr></thead><tbody>' +
                    models.map(m => {
                        const arch = m.architecture === 'moe' ? '<span style="font-size:9px;padding:1px 4px;border-radius:3px;background:var(--purple-bg);color:var(--purple);">MoE</span>' : '';
                        return '<tr><td style="padding:4px 6px;">' + m.name + ' ' + arch + '</td><td style="text-align:right;padding:4px 6px;font-weight:600;">$' + m.price_train.toFixed(2) + '</td><td style="text-align:right;padding:4px 6px;">$' + m.price_sample.toFixed(2) + '</td></tr>';
                    }).join('') + '</tbody></table>';
            });
        }

        document.getElementById('tk-estimate-btn').addEventListener('click', () => {
            const modelId = document.getElementById('tk-model').value;
            // Find model key from ID
                const m = models.find(x => x.id === modelId);
                const key = m ? m.key : '';
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_key: key, dataset_tokens: 100000, epochs: parseInt(document.getElementById('tk-epochs').value)})
                }).then(r => r.json()).then(est => {
                    document.getElementById('tk-cost-result').innerHTML =
                        '<div style="padding:12px;background:var(--blue-bg);border-radius:8px;font-size:12px;color:var(--blue);">' +
                        '&#128176; Estimated cost for 100K tokens: <strong>$' + (est.total_estimated_usd || 0).toFixed(4) + '</strong>' +
                        ' &bull; Per epoch: $' + (est.per_epoch_usd || 0).toFixed(4) +
                        ' &bull; Train: $' + (est.price_per_1m_tokens || 0).toFixed(2) + '/1M tokens</div>';
                });
            });
        });

        document.getElementById('tk-train-btn').addEventListener('click', () => {
            const req = {
                base_model: document.getElementById('tk-model').value,
                lora_rank: parseInt(document.getElementById('tk-rank').value),
                learning_rate: parseFloat(document.getElementById('tk-lr').value),
                num_epochs: parseInt(document.getElementById('tk-epochs').value),
                batch_size: parseInt(document.getElementById('tk-batch').value),
                loss_function: document.getElementById('tk-loss').value,
                train_mlp: document.getElementById('tk-mlp').checked,
                train_attn: document.getElementById('tk-attn').checked,
                train_unembed: document.getElementById('tk-unembed').checked,
            };
                .then(r => r.json()).then(job => {
                });
        });

                const el = document.getElementById('tk-jobs');
                el.innerHTML = jobs.map(j => {
                    const colors = { completed: 'var(--green)', training: 'var(--blue)', failed: '#ef4444', pending: 'var(--gray-500)' };
                    const c = colors[j.status] || 'var(--gray-500)';
                    return '<div style="padding:12px 14px;border:1px solid var(--gray-100);border-radius:8px;margin-bottom:6px;display:flex;align-items:center;gap:10px;">' +
                        '<span style="font-weight:600;font-size:13px;color:var(--navy);">' + j.name + '</span>' +
                        '<span style="font-size:11px;color:var(--gray-500);">' + (j.config.base_model || '').split('/').pop() + '</span>' +
                        '<span style="font-size:10px;padding:2px 8px;border-radius:4px;background:' + c + '15;color:' + c + ';font-weight:600;margin-left:auto;">' + j.status + '</span></div>';
                }).join('');
            });
        }

        document.getElementById('ds-split-btn').addEventListener('click', () => {
            if (!currentDatasetId) return;
            const train = parseInt(document.getElementById('ds-train').value) / 100;
            const val = parseInt(document.getElementById('ds-val').value) / 100;
            const test = parseInt(document.getElementById('ds-test').value) / 100;
            fetch('/api/finetune/datasets/' + currentDatasetId + '/split', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({train_ratio: train, val_ratio: val, test_ratio: test, seed: 42})
            }).then(r => r.json()).then(res => {
                document.getElementById('ds-split-result').innerHTML =
                    '<div style="padding:12px;background:var(--green-bg);border-radius:8px;color:var(--green);font-weight:500;">' +
                    '&#9989; Split complete! Train: ' + res.train_count + ' &bull; Val: ' + res.val_count + ' &bull; Test: ' + res.test_count + '</div>';
                loadDatasetList();
            });
        });
    </script>
</body>
</html>"""


from typing import Any
from pydantic import BaseModel as _BaseModel


class _ChatRequest(_BaseModel):
    message: str


def create_app() -> Any:
    """Create the FastAPI application for Duxx AI Studio."""
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Duxx AI Studio", version="0.1.0", docs_url="/api/docs", redoc_url="/api/redoc")

    # In-memory state
    state: dict[str, Any] = {"agent": None, "tracer": None}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        from duxx_ai.studio.docs import DOCS_TEMPLATE
        return DOCS_TEMPLATE

    @app.get("/docs", response_class=HTMLResponse)
    async def docs() -> str:
        from duxx_ai.studio.docs import DOCS_TEMPLATE
        return DOCS_TEMPLATE

    @app.get("/studio", response_class=HTMLResponse)
    async def studio_page() -> str:
        return HTML_TEMPLATE

    @app.post("/api/chat")
    async def chat(req: _ChatRequest) -> dict[str, Any]:
        if state["agent"] is None:
            from duxx_ai.core.agent import Agent, AgentConfig
            from duxx_ai.tools.builtin import get_builtin_tools
            from duxx_ai.observability.tracer import Tracer

            tracer = Tracer()
            state["tracer"] = tracer
            agent = Agent(
                config=AgentConfig(name="studio-agent"),
                tools=get_builtin_tools(["python_exec", "calculator", "read_file"]),
                tracer=tracer,
            )
            state["agent"] = agent

        agent = state["agent"]
        response = await agent.run(req.message)

        metrics = {
            "tokens": agent.state.total_tokens,
            "latency": 0,
            "cost": agent.state.total_cost,
        }

        tracer = state["tracer"]
        traces = []
        if tracer and tracer.traces:
            last_trace = tracer.traces[-1]
            traces = [
                {"name": s.name, "duration": round(s.duration_ms, 1)}
                for s in last_trace.spans
            ]

        return {"response": response, "metrics": metrics, "traces": traces}

    @app.get("/api/tools")
    async def list_tools() -> list[dict[str, str]]:
        from duxx_ai.tools.builtin import BUILTIN_TOOLS
        return [{"name": t.name, "description": t.description} for t in BUILTIN_TOOLS.values()]

    @app.get("/api/agents")
    async def list_agents() -> list[dict[str, Any]]:
        if state["agent"]:
            a = state["agent"]
            return [{"name": a.name, "status": a.state.status, "tokens": a.state.total_tokens}]
        return []

    @app.get("/api/templates")
    async def list_templates() -> list[dict[str, Any]]:
        from duxx_ai.templates import TEMPLATES
        result = []
        for name, cls in TEMPLATES.items():
            info = cls.info()
            result.append({
                "name": info["name"],
                "category": info["category"],
                "description": info["description"],
                "tools": info.get("default_tools", []),
            })
        return result

    @app.get("/api/governance")
    async def governance_info() -> dict[str, Any]:
        return {
            "guardrails": [
                {"name": "PII Filter", "type": "input"},
                {"name": "Prompt Injection", "type": "input"},
                {"name": "Content Filter", "type": "output"},
                {"name": "Hallucination Check", "type": "output"},
                {"name": "Token Budget", "type": "budget"},
            ],
            "roles": ["Admin", "Developer", "Operator", "Viewer"],
        }

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    return app
