"""Duxx AI Documentation — Complete enterprise SDK documentation site."""

from __future__ import annotations

DOCS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Duxx AI Documentation</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --navy: #0c1b3a;
            --navy-light: #14284b;
            --navy-50: #eef1f6;
            --pink: #d6336c;
            --pink-light: #f06595;
            --pink-bg: #fdf2f6;
            --pink-50: #fff0f6;
            --black: #1a1a1a;
            --gray-900: #212529;
            --gray-800: #343a40;
            --gray-700: #495057;
            --gray-600: #6c757d;
            --gray-500: #888;
            --gray-400: #adb5bd;
            --gray-300: #ced4da;
            --gray-200: #e5e7eb;
            --gray-100: #f3f4f6;
            --gray-50: #f9fafb;
            --white: #ffffff;
            --green: #16a34a;
            --green-bg: #f0fdf4;
            --blue: #2563eb;
            --blue-bg: #eff6ff;
            --orange: #e8590c;
            --orange-bg: #fff4e6;
            --purple: #7c3aed;
            --purple-bg: #f5f3ff;
            --radius: 8px;
            --radius-lg: 12px;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
            --sidebar-w: 280px;
            --toc-w: 220px;
            --header-h: 60px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--white); color: var(--gray-800);
            line-height: 1.7; font-size: 15px;
        }

        /* ── Animations ── */
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-12px); }
        }
        @keyframes float-slow {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-8px) rotate(3deg); }
        }
        @keyframes pulse-glow {
            0%, 100% { opacity: 0.4; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.05); }
        }
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes fade-in-up {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slide-in-left {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes particle-drift {
            0% { transform: translate(0, 0) scale(1); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translate(var(--dx, 80px), var(--dy, -120px)) scale(0.3); opacity: 0; }
        }
        @keyframes border-glow {
            0%, 100% { border-color: var(--gray-200); }
            50% { border-color: var(--pink-light); }
        }

        .animate-in { animation: fade-in-up 0.5s ease both; }

        /* ── Top Header ── */
        .top-header {
            position: fixed; top: 0; left: 0; right: 0; z-index: 100;
            height: var(--header-h);
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(229,231,235,0.6);
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .top-header-brand { display: flex; align-items: center; gap: 10px; text-decoration: none; }
        .top-header-brand .logo {
            width: 32px; height: 32px; border-radius: 8px;
            background: linear-gradient(-45deg, var(--navy), var(--pink), #1a3a6e, var(--pink-light));
            background-size: 300% 300%;
            animation: gradient-shift 6s ease infinite;
            display: flex; align-items: center; justify-content: center;
            color: var(--white); font-weight: 700; font-size: 13px;
            transition: transform 0.3s;
        }
        .top-header-brand:hover .logo { transform: rotate(8deg) scale(1.1); }
        .top-header-brand h1 { font-size: 18px; font-weight: 700; color: var(--navy); }
        .top-header-brand h1 span { color: var(--pink); }
        .top-header-badge {
            font-size: 11px; background: var(--pink-bg); color: var(--pink);
            padding: 2px 8px; border-radius: 4px; font-weight: 600; margin-left: 8px;
        }
        .top-header-nav { display: flex; align-items: center; gap: 6px; }
        .top-header-nav a {
            font-size: 13px; font-weight: 500; color: var(--gray-600);
            text-decoration: none; padding: 6px 14px; border-radius: 6px;
            transition: all 0.15s;
        }
        .top-header-nav a:hover { color: var(--navy); background: var(--gray-100); }
        .top-header-nav a.active { color: var(--pink); background: var(--pink-bg); }
        .top-header-nav .btn-studio {
            background: var(--navy); color: var(--white); font-weight: 600;
            padding: 7px 18px; border-radius: 6px;
        }
        .top-header-nav .btn-studio:hover { background: var(--navy-light); color: var(--white); }

        /* ── Layout ── */
        .docs-layout {
            display: flex; margin-top: var(--header-h); min-height: calc(100vh - var(--header-h));
        }

        /* ── Sidebar ── */
        .docs-sidebar {
            width: var(--sidebar-w); position: fixed; top: var(--header-h); bottom: 0;
            left: 0; background: var(--gray-50); border-right: 1px solid var(--gray-200);
            overflow-y: auto; padding: 20px 0; z-index: 50;
        }
        .docs-sidebar::-webkit-scrollbar { width: 4px; }
        .docs-sidebar::-webkit-scrollbar-thumb { background: var(--gray-300); border-radius: 2px; }
        .sidebar-section { margin-bottom: 8px; }
        .sidebar-section-title {
            font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1.2px; color: var(--gray-500); padding: 12px 24px 6px;
        }
        .sidebar-link {
            display: block; padding: 7px 24px 7px 28px; font-size: 13px;
            color: var(--gray-700); text-decoration: none; border-left: 3px solid transparent;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer; font-weight: 450;
        }
        .sidebar-link:hover { color: var(--navy); background: var(--white); padding-left: 32px; }
        .sidebar-link.active {
            color: var(--pink); border-left-color: var(--pink);
            background: var(--pink-bg); font-weight: 600;
        }
        .sidebar-link.sub { padding-left: 42px; font-size: 12.5px; color: var(--gray-600); }

        /* ── Content ── */
        .docs-content {
            margin-left: var(--sidebar-w); flex: 1; padding: 40px 56px 80px;
            max-width: 860px; min-height: 100vh;
        }

        /* ── TOC ── */
        .docs-toc {
            position: fixed; top: var(--header-h); right: 0; width: var(--toc-w);
            padding: 28px 20px; overflow-y: auto; bottom: 0;
        }
        .docs-toc h4 {
            font-size: 11px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1px; color: var(--gray-500); margin-bottom: 12px;
        }
        .docs-toc a {
            display: block; font-size: 12px; color: var(--gray-500);
            text-decoration: none; padding: 4px 0; border-left: 2px solid var(--gray-200);
            padding-left: 12px; transition: all 0.12s;
        }
        .docs-toc a:hover { color: var(--navy); border-left-color: var(--navy); }
        .docs-toc a.active { color: var(--pink); border-left-color: var(--pink); font-weight: 600; }

        /* ── Typography ── */
        .docs-content h1 {
            font-size: 32px; font-weight: 800; color: var(--navy);
            margin-bottom: 12px; letter-spacing: -0.5px; line-height: 1.2;
        }
        .docs-content h2 {
            font-size: 22px; font-weight: 700; color: var(--navy);
            margin-top: 48px; margin-bottom: 16px; padding-bottom: 8px;
            border-bottom: 1px solid var(--gray-200); letter-spacing: -0.3px;
        }
        .docs-content h3 {
            font-size: 17px; font-weight: 600; color: var(--navy);
            margin-top: 32px; margin-bottom: 12px;
        }
        .docs-content p { margin-bottom: 16px; color: var(--gray-700); }
        .docs-content ul, .docs-content ol { margin-bottom: 16px; padding-left: 24px; }
        .docs-content li { margin-bottom: 6px; color: var(--gray-700); }
        .docs-content a { color: var(--pink); text-decoration: none; font-weight: 500; }
        .docs-content a:hover { text-decoration: underline; }
        .lead { font-size: 17px; color: var(--gray-600); line-height: 1.7; margin-bottom: 32px; }

        /* ── Code Blocks ── */
        code {
            font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
            font-size: 13px;
        }
        :not(pre) > code {
            background: var(--gray-100); color: var(--pink);
            padding: 2px 6px; border-radius: 4px; font-size: 13px;
        }
        pre {
            background: var(--navy); color: #e0e7ff; border-radius: var(--radius-lg);
            padding: 20px 24px; margin-bottom: 24px; overflow-x: auto;
            line-height: 1.6; position: relative;
        }
        pre code { color: #e0e7ff; background: none; padding: 0; font-size: 13px; }
        pre .comment { color: #6b7994; }
        pre .keyword { color: #f06595; }
        pre .string { color: #69db7c; }
        pre .function { color: #74c0fc; }
        pre .decorator { color: #ffd43b; }
        pre .number { color: #ffa94d; }
        .code-header {
            display: flex; justify-content: space-between; align-items: center;
            background: #081428; padding: 8px 16px; border-radius: var(--radius-lg) var(--radius-lg) 0 0;
            margin-bottom: 0;
        }
        .code-header span { font-size: 12px; color: #6b7994; font-weight: 500; }
        .code-header + pre { border-radius: 0 0 var(--radius-lg) var(--radius-lg); margin-top: 0; }

        /* ── Callouts ── */
        .callout {
            border-radius: var(--radius-lg); padding: 16px 20px; margin-bottom: 24px;
            display: flex; gap: 12px; font-size: 14px; border-left: 4px solid;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .callout:hover { transform: translateX(4px); box-shadow: var(--shadow-sm); }
        .callout-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
        .callout-info { background: var(--blue-bg); border-color: var(--blue); }
        .callout-warn { background: var(--orange-bg); border-color: var(--orange); }
        .callout-tip { background: var(--green-bg); border-color: var(--green); }
        .callout-note { background: var(--purple-bg); border-color: var(--purple); }
        .callout p { margin-bottom: 0; }
        .callout strong { color: var(--navy); }

        /* ── Feature Cards ── */
        .feature-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px;
        }
        .feature-card {
            background: rgba(249,250,251,0.7); border: 1px solid var(--gray-200);
            border-radius: 14px; padding: 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer;
            backdrop-filter: blur(6px); position: relative; overflow: hidden;
        }
        .feature-card::before {
            content: ''; position: absolute; inset: 0; border-radius: 14px;
            background: linear-gradient(135deg, rgba(214,51,108,0.03), rgba(37,99,235,0.03));
            opacity: 0; transition: opacity 0.3s;
        }
        .feature-card:hover {
            border-color: var(--pink); transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(12,27,58,0.1);
        }
        .feature-card:hover::before { opacity: 1; }
        .feature-card h4 { font-size: 15px; font-weight: 600; color: var(--navy); margin-bottom: 6px; position: relative; }
        .feature-card p { font-size: 13px; color: var(--gray-600); margin-bottom: 0; line-height: 1.5; position: relative; }
        .feature-card .icon {
            font-size: 28px; margin-bottom: 12px; display: inline-block;
            animation: float-slow 4s ease-in-out infinite;
            animation-delay: var(--delay, 0s);
        }

        /* ── Tabs ── */
        .tabs { display: flex; gap: 0; border-bottom: 2px solid var(--gray-200); margin-bottom: 20px; }
        .tab {
            padding: 10px 20px; font-size: 13px; font-weight: 500;
            color: var(--gray-500); cursor: pointer; border-bottom: 2px solid transparent;
            margin-bottom: -2px; transition: all 0.15s;
        }
        .tab:hover { color: var(--navy); }
        .tab.active { color: var(--pink); border-bottom-color: var(--pink); font-weight: 600; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* ── Tables ── */
        table {
            width: 100%; border-collapse: collapse; margin-bottom: 24px;
            font-size: 14px; border-radius: var(--radius); overflow: hidden;
        }
        th {
            background: var(--gray-50); text-align: left; padding: 10px 16px;
            font-weight: 600; color: var(--navy); border-bottom: 2px solid var(--gray-200);
            font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
        }
        td { padding: 10px 16px; border-bottom: 1px solid var(--gray-100); color: var(--gray-700); }
        tr:hover td { background: var(--gray-50); }

        /* ── Badges ── */
        .badge {
            display: inline-block; font-size: 11px; font-weight: 600;
            padding: 2px 8px; border-radius: 4px; margin-left: 6px;
        }
        .badge-new { background: var(--green-bg); color: var(--green); }
        .badge-enterprise { background: var(--purple-bg); color: var(--purple); }
        .badge-beta { background: var(--orange-bg); color: var(--orange); }

        /* ── Hero ── */
        .hero {
            background: linear-gradient(-45deg, #0c1b3a, #14284b, #1a3a6e, #0f2347);
            background-size: 400% 400%;
            animation: gradient-shift 12s ease infinite;
            border-radius: 20px; padding: 56px 48px; margin-bottom: 40px;
            color: var(--white); position: relative; overflow: hidden;
        }
        .hero::before {
            content: ''; position: absolute; top: -60%; right: -25%;
            width: 500px; height: 500px; border-radius: 50%;
            background: radial-gradient(circle, rgba(214,51,108,0.25), transparent 70%);
            animation: pulse-glow 6s ease-in-out infinite;
        }
        .hero::after {
            content: ''; position: absolute; bottom: -40%; left: -15%;
            width: 350px; height: 350px; border-radius: 50%;
            background: radial-gradient(circle, rgba(37,99,235,0.15), transparent 70%);
            animation: pulse-glow 8s ease-in-out infinite 2s;
        }
        .hero-particles {
            position: absolute; inset: 0; overflow: hidden; pointer-events: none;
        }
        .hero-particle {
            position: absolute; width: 4px; height: 4px; border-radius: 50%;
            background: rgba(255,255,255,0.3);
            animation: particle-drift var(--duration, 8s) linear infinite;
            animation-delay: var(--delay, 0s);
        }
        .hero h1 {
            color: var(--white); font-size: 40px; margin-bottom: 14px;
            position: relative; letter-spacing: -0.5px;
            animation: fade-in-up 0.6s ease both;
        }
        .hero h1 span { color: var(--pink-light); }
        .hero p {
            color: rgba(255,255,255,0.8); font-size: 17px; max-width: 600px;
            position: relative; margin-bottom: 28px;
            animation: fade-in-up 0.6s ease 0.15s both;
        }
        .hero-actions {
            display: flex; gap: 12px; position: relative;
            animation: fade-in-up 0.6s ease 0.3s both;
        }
        .hero-btn {
            display: inline-block; padding: 12px 28px; border-radius: 10px;
            font-weight: 600; font-size: 14px; text-decoration: none;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer;
        }
        .hero-btn:hover { transform: translateY(-2px); }
        .hero-btn-primary {
            background: var(--pink); color: #ffffff !important; border: none;
            box-shadow: 0 4px 15px rgba(214,51,108,0.4);
        }
        .hero-btn-primary:hover {
            background: var(--pink-light); text-decoration: none; color: #ffffff !important;
            box-shadow: 0 6px 25px rgba(214,51,108,0.5);
        }
        .hero-btn-secondary {
            background: rgba(255,255,255,0.1); color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.25);
            backdrop-filter: blur(8px);
        }
        .hero-btn-secondary:hover {
            background: rgba(255,255,255,0.2); text-decoration: none; color: #ffffff !important;
            border-color: rgba(255,255,255,0.4);
        }

        /* ── Install Block ── */
        .install-block {
            background: linear-gradient(90deg, var(--navy), #14284b);
            border-radius: var(--radius-lg);
            padding: 16px 24px; font-family: 'JetBrains Mono', monospace;
            color: #e0e7ff; font-size: 14px; margin-bottom: 24px;
            display: flex; justify-content: space-between; align-items: center;
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.3s;
        }
        .install-block:hover {
            box-shadow: 0 4px 20px rgba(12,27,58,0.2);
            border-color: rgba(214,51,108,0.3);
        }
        .install-block .dollar { color: var(--pink-light); margin-right: 8px; }

        /* ── Page sections ── */
        .page { display: none; }
        .page.active { display: block; }

        /* ── Architecture Diagram ── */
        .arch-diagram {
            background: var(--gray-50); border: 1px solid var(--gray-200);
            border-radius: var(--radius-lg); padding: 32px; margin: 24px 0;
            font-family: 'JetBrains Mono', monospace; font-size: 13px;
            line-height: 1.5; color: var(--navy); white-space: pre; overflow-x: auto;
        }

        /* ── Footer ── */
        .docs-footer {
            margin-top: 64px; padding-top: 32px; border-top: 1px solid var(--gray-200);
            display: flex; justify-content: space-between;
        }
        .docs-footer a {
            display: flex; flex-direction: column; gap: 4px; padding: 16px 20px;
            background: var(--gray-50); border: 1px solid var(--gray-200);
            border-radius: var(--radius-lg); text-decoration: none; transition: all 0.15s;
            min-width: 180px;
        }
        .docs-footer a:hover { border-color: var(--pink); box-shadow: var(--shadow-sm); text-decoration: none; }
        .docs-footer .label { font-size: 12px; color: var(--gray-500); font-weight: 500; }
        .docs-footer .title { font-size: 15px; color: var(--navy); font-weight: 600; }

        /* ── Responsive ── */
        @media (max-width: 1200px) { .docs-toc { display: none; } }
        @media (max-width: 900px) {
            .docs-sidebar { display: none; }
            .docs-content { margin-left: 0; padding: 24px; }
            .feature-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<!-- ── Top Header ── -->
<div class="top-header">
    <a href="/docs" class="top-header-brand">
        <div class="logo">B</div>
        <h1>Duxx <span>AI</span></h1>
        <span class="top-header-badge">v0.1.0</span>
    </a>
    <div class="top-header-nav">
        <a href="/docs" class="active">Docs</a>
        <a href="/docs#api-reference" onclick="navigateTo('api-reference')">API</a>
        <a href="https://github.com/bankyresearch/duxx-ai" target="_blank">GitHub</a>
        <a href="/studio" class="btn-studio">Studio</a>
    </div>
</div>

<div class="docs-layout">

<!-- ── Sidebar ── -->
<nav class="docs-sidebar">
    <div class="sidebar-section">
        <div class="sidebar-section-title">Getting Started</div>
        <a class="sidebar-link active" onclick="navigateTo('overview')">Overview</a>
        <a class="sidebar-link" onclick="navigateTo('installation')">Installation</a>
        <a class="sidebar-link" onclick="navigateTo('quickstart')">Quick Start</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Core Concepts</div>
        <a class="sidebar-link" onclick="navigateTo('agents')">Agents</a>
        <a class="sidebar-link" onclick="navigateTo('agent-advanced')">Advanced Agent</a>
        <a class="sidebar-link" onclick="navigateTo('tools')">Tools</a>
        <a class="sidebar-link" onclick="navigateTo('parsers')">Output Parsers</a>
        <a class="sidebar-link" onclick="navigateTo('providers')">LLM Providers</a>
        <a class="sidebar-link" onclick="navigateTo('llm-advanced')">Cache &amp; Rate Limit</a>
        <a class="sidebar-link" onclick="navigateTo('messages')">Messages</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Orchestration</div>
        <a class="sidebar-link" onclick="navigateTo('graph')">Graph Workflows</a>
        <a class="sidebar-link" onclick="navigateTo('graph-advanced')">Advanced Graph</a>
        <a class="sidebar-link" onclick="navigateTo('crew')">Multi-Agent Crews</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">RAG Pipeline</div>
        <a class="sidebar-link" onclick="navigateTo('rag')">RAG Overview</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Fine-Tuning</div>
        <a class="sidebar-link" onclick="navigateTo('finetune')">Pipeline Overview</a>
        <a class="sidebar-link" onclick="navigateTo('lora')">LoRA / QLoRA</a>
        <a class="sidebar-link" onclick="navigateTo('dataset')">Dataset Preparation</a>
        <a class="sidebar-link" onclick="navigateTo('connectors')">Data Connectors</a>
        <a class="sidebar-link" onclick="navigateTo('tinker')">Tinker Cloud</a>
        <a class="sidebar-link" onclick="navigateTo('model-storage')">Model Storage</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Governance</div>
        <a class="sidebar-link" onclick="navigateTo('guardrails')">Guardrails</a>
        <a class="sidebar-link" onclick="navigateTo('rbac')">RBAC</a>
        <a class="sidebar-link" onclick="navigateTo('audit')">Audit Logging</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Memory</div>
        <a class="sidebar-link" onclick="navigateTo('memory')">5-Tier Memory</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Routing</div>
        <a class="sidebar-link" onclick="navigateTo('router')">Adaptive Router</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Observability</div>
        <a class="sidebar-link" onclick="navigateTo('tracing')">Tracing</a>
        <a class="sidebar-link" onclick="navigateTo('evaluation')">Evaluation</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Agent Patterns</div>
        <a class="sidebar-link" onclick="navigateTo('react')">ReAct Agent</a>
        <a class="sidebar-link" onclick="navigateTo('handoffs')">Agent Handoffs</a>
        <a class="sidebar-link" onclick="navigateTo('self-improving')">Self-Improving Agent</a>
        <a class="sidebar-link" onclick="navigateTo('teachable')">Teachable Agent</a>
        <a class="sidebar-link" onclick="navigateTo('eval-optimizer')">Evaluator-Optimizer</a>
        <a class="sidebar-link" onclick="navigateTo('orchestrator-worker')">Orchestrator-Worker</a>
        <a class="sidebar-link" onclick="navigateTo('parallel-guardrails')">Parallel Guardrails</a>
        <a class="sidebar-link" onclick="navigateTo('agentic-rag')">Agentic RAG</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Deep Agent</div>
        <a class="sidebar-link" onclick="navigateTo('deep-agent')">Deep Agent Architecture</a>
        <a class="sidebar-link" onclick="navigateTo('vfs')">Virtual File System</a>
        <a class="sidebar-link" onclick="navigateTo('planning')">Planning Tool</a>
        <a class="sidebar-link" onclick="navigateTo('graph-analytics')">Graph Analytics</a>
        <a class="sidebar-link" onclick="navigateTo('a2a')">A2A Protocol</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">LangGraph Compatible</div>
        <a class="sidebar-link" onclick="navigateTo('lg-compile')">compile() &amp; Command</a>
        <a class="sidebar-link" onclick="navigateTo('lg-interrupt')">Interrupts &amp; HITL</a>
        <a class="sidebar-link" onclick="navigateTo('lg-checkpoints')">Checkpointers</a>
        <a class="sidebar-link" onclick="navigateTo('lg-task')">@task &amp; Durable Execution</a>
        <a class="sidebar-link" onclick="navigateTo('lg-streaming')">Streaming Modes</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Importers</div>
        <a class="sidebar-link" onclick="navigateTo('n8n-import')">n8n Workflow Import</a>
        <a class="sidebar-link" onclick="navigateTo('mcp')">MCP Integration</a>
        <a class="sidebar-link" onclick="navigateTo('middleware')">Middleware</a>

        <h3>Integration Catalog (490)</h3>
        <a class="sidebar-link" onclick="navigateTo('integrations')">Overview</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-chat-detail')">Chat Models (72)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-embed-detail')">Embeddings (86)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-vs-detail')">Vector Stores (153)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-loader-detail')">Doc Loaders (63)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-ret-detail')">Retrievers (94)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-check-detail')">Checkpointers (8)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-parser-detail')">Output Parsers (10)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-mw-detail')">Middleware (4)</a>
        <a class="sidebar-link sub" onclick="navigateTo('int-contextual')">Contextual Retrieval</a>

        <h3>FlowGraph Engine</h3>
        <a class="sidebar-link" onclick="navigateTo('flowgraph')">FlowGraph Overview</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Templates</div>
        <a class="sidebar-link" onclick="navigateTo('templates')">14 Enterprise Agents</a>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Reference</div>
        <a class="sidebar-link" onclick="navigateTo('api-reference')">API Reference</a>
        <a class="sidebar-link" onclick="navigateTo('cli-ref')">CLI Reference</a>
    </div>
</nav>

<!-- ── Main Content ── -->
<main class="docs-content">

<!-- ════════════════ OVERVIEW ════════════════ -->
<div id="page-overview" class="page active">
<div class="hero">
    <div class="hero-particles" id="hero-particles"></div>
    <h1>Duxx <span>AI</span> Enterprise SDK</h1>
    <p>Build, fine-tune, orchestrate, and govern AI agents at scale. The only open-source SDK that unifies agentic AI with fine-tuning pipelines and enterprise governance.</p>
    <div class="hero-actions">
        <a class="hero-btn hero-btn-primary" onclick="navigateTo('quickstart')">Get Started</a>
        <a class="hero-btn hero-btn-secondary" href="https://github.com/bankyresearch/duxx-ai" target="_blank">View on GitHub</a>
    </div>
</div>

<div class="install-block">
    <span><span class="dollar">$</span> pip install duxx_ai</span>
</div>

<h2 id="why-duxx_ai">Why Duxx AI?</h2>
<p>Existing frameworks force you to choose: agent orchestration <em>or</em> model fine-tuning. Duxx AI delivers both in a single, enterprise-ready platform.</p>

<div class="feature-grid">
    <div class="feature-card" onclick="navigateTo('agents')" style="animation: fade-in-up 0.5s ease 0.1s both">
        <div class="icon" style="--delay:0s">&#129302;</div>
        <h4>Agent SDK</h4>
        <p>Multi-provider agents with tool use, multi-turn conversation, and configurable LLM backends (OpenAI, Anthropic, local models).</p>
    </div>
    <div class="feature-card" onclick="navigateTo('graph')" style="animation: fade-in-up 0.5s ease 0.2s both">
        <div class="icon" style="--delay:0.5s">&#128300;</div>
        <h4>Graph Orchestration</h4>
        <p>DAG-based workflows with conditional routing, parallel branches, state reducers, human-in-the-loop, and map-reduce.</p>
    </div>
    <div class="feature-card" onclick="navigateTo('finetune')" style="animation: fade-in-up 0.5s ease 0.3s both">
        <div class="icon" style="--delay:1s">&#9881;&#65039;</div>
        <h4>Fine-Tuning Pipeline</h4>
        <p>Integrated Unsloth/PEFT training. Convert agent traces to training data. LoRA, QLoRA, GGUF export.</p>
    </div>
    <div class="feature-card" onclick="navigateTo('guardrails')" style="animation: fade-in-up 0.5s ease 0.4s both">
        <div class="icon" style="--delay:1.5s">&#128737;&#65039;</div>
        <h4>Enterprise Governance</h4>
        <p>Built-in guardrails, RBAC with role inheritance, audit logging, and policy enforcement.</p>
    </div>
    <div class="feature-card" onclick="navigateTo('router')" style="animation: fade-in-up 0.5s ease 0.5s both">
        <div class="icon" style="--delay:2s">&#128268;</div>
        <h4>Adaptive Router</h4>
        <p>Complexity-based model routing. Send simple tasks to cheap models, complex tasks to powerful ones. Budget enforcement.</p>
    </div>
    <div class="feature-card" onclick="navigateTo('memory')" style="animation: fade-in-up 0.5s ease 0.6s both">
        <div class="icon" style="--delay:2.5s">&#129504;</div>
        <h4>5-Tier Memory</h4>
        <p>Working, episodic, semantic, procedural, and shared memory with TTL, LRU eviction, and cross-agent coordination.</p>
    </div>
</div>

<h2 id="architecture">Architecture</h2>
<div class="arch-diagram">duxx_ai/
  core/           Agent, Tool, Message, LLM provider abstractions
  orchestration/  Graph engine + Crew-based multi-agent coordination
  finetune/       Fine-tuning pipeline (Unsloth / PEFT integration)
  governance/     Guardrails, RBAC, audit logging
  router/         Adaptive model routing with complexity estimation
  memory/         5-tier memory system (working, episodic, semantic, procedural, shared)
  observability/  OpenTelemetry-compatible tracing
  tools/          Built-in tools (code exec, file ops, web requests)
  studio/         FastAPI web UI for agent management
  cli/            Command-line interface</div>

<h2 id="comparison">How Duxx AI Compares</h2>
<table>
    <thead><tr><th>Feature</th><th>Duxx AI</th><th>LangChain</th><th>CrewAI</th><th>Unsloth</th></tr></thead>
    <tbody>
        <tr><td>Agent SDK</td><td><strong style="color:var(--green)">&#10003;</strong></td><td>&#10003;</td><td>&#10003;</td><td>&#10007;</td></tr>
        <tr><td>Graph Orchestration</td><td><strong style="color:var(--green)">&#10003;</strong></td><td>&#10003; (LangGraph)</td><td>&#10007;</td><td>&#10007;</td></tr>
        <tr><td>Multi-Agent Crews</td><td><strong style="color:var(--green)">&#10003;</strong></td><td>&#10007;</td><td>&#10003;</td><td>&#10007;</td></tr>
        <tr><td>Fine-Tuning Pipeline</td><td><strong style="color:var(--green)">&#10003;</strong></td><td>&#10007;</td><td>&#10007;</td><td>&#10003;</td></tr>
        <tr><td>Guardrails &amp; RBAC</td><td><strong style="color:var(--green)">&#10003;</strong></td><td>&#10007;</td><td>&#10007;</td><td>&#10007;</td></tr>
        <tr><td>Adaptive Routing</td><td><strong style="color:var(--green)">&#10003;</strong></td><td>&#10007;</td><td>&#10007;</td><td>&#10007;</td></tr>
        <tr><td>Memory System</td><td><strong style="color:var(--green)">5-tier</strong></td><td>Basic</td><td>Basic</td><td>&#10007;</td></tr>
        <tr><td>Observability</td><td><strong style="color:var(--green)">&#10003;</strong></td><td>&#10003;</td><td>&#10007;</td><td>&#10007;</td></tr>
    </tbody>
</table>

<div class="docs-footer">
    <div></div>
    <a onclick="navigateTo('installation')">
        <span class="label">Next</span>
        <span class="title">Installation &#8594;</span>
    </a>
</div>
</div>

<!-- ════════════════ INSTALLATION ════════════════ -->
<div id="page-installation" class="page">
<h1>Installation</h1>
<p class="lead">Install Duxx AI and its optional dependencies for fine-tuning, the Studio web UI, or the full enterprise suite.</p>

<h2 id="install-core">Core SDK</h2>
<p>The core SDK includes agents, tools, orchestration, governance, memory, routing, and observability.</p>
<div class="install-block"><span><span class="dollar">$</span> pip install duxx_ai</span></div>

<h2 id="install-extras">Optional Extras</h2>
<div class="code-header"><span>Fine-tuning support (Unsloth + PEFT)</span></div>
<pre><code>pip install duxx_ai[finetune]</code></pre>

<div class="code-header"><span>Studio web UI (FastAPI)</span></div>
<pre><code>pip install duxx_ai[studio]</code></pre>

<div class="code-header"><span>Everything</span></div>
<pre><code>pip install duxx_ai[all]</code></pre>

<h2 id="install-requirements">Requirements</h2>
<table>
    <thead><tr><th>Dependency</th><th>Version</th><th>Purpose</th></tr></thead>
    <tbody>
        <tr><td><code>Python</code></td><td>&gt;= 3.10</td><td>Runtime</td></tr>
        <tr><td><code>pydantic</code></td><td>&gt;= 2.0</td><td>Data models</td></tr>
        <tr><td><code>httpx</code></td><td>&gt;= 0.25</td><td>HTTP client for LLM APIs</td></tr>
        <tr><td><code>tiktoken</code></td><td>&gt;= 0.5</td><td>Token counting</td></tr>
        <tr><td><code>opentelemetry-sdk</code></td><td>&gt;= 1.20</td><td>Tracing</td></tr>
        <tr><td><code>rich</code></td><td>&gt;= 13.0</td><td>Terminal output</td></tr>
        <tr><td><code>click</code></td><td>&gt;= 8.0</td><td>CLI framework</td></tr>
    </tbody>
</table>

<div class="callout callout-tip">
    <span class="callout-icon">&#128161;</span>
    <div><strong>GPU Fine-Tuning:</strong> For fine-tuning, you need a CUDA-capable GPU with at least 8GB VRAM. QLoRA with 4-bit quantization can fine-tune 7B models on a single 16GB GPU.</div>
</div>

<h2 id="install-verify">Verify Installation</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">import</span> duxx_ai
<span class="function">print</span>(duxx_ai.__version__)  <span class="comment"># 0.1.0</span></code></pre>

<div class="docs-footer">
    <a onclick="navigateTo('overview')">
        <span class="label">&#8592; Previous</span>
        <span class="title">Overview</span>
    </a>
    <a onclick="navigateTo('quickstart')">
        <span class="label">Next</span>
        <span class="title">Quick Start &#8594;</span>
    </a>
</div>
</div>

<!-- ════════════════ QUICK START ════════════════ -->
<div id="page-quickstart" class="page">
<h1>Quick Start</h1>
<p class="lead">Build your first AI agent in under 5 minutes. This guide covers creating an agent, adding tools, and running a conversation.</p>

<h2 id="qs-agent">1. Create an Agent</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">import</span> asyncio
<span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Agent, AgentConfig
<span class="keyword">from</span> duxx_ai.core.llm <span class="keyword">import</span> LLMConfig
<span class="keyword">from</span> duxx_ai.tools.builtin <span class="keyword">import</span> get_builtin_tools

agent = Agent(
    config=AgentConfig(
        name=<span class="string">"assistant"</span>,
        system_prompt=<span class="string">"You are a helpful assistant."</span>,
        llm=LLMConfig(provider=<span class="string">"openai"</span>, model=<span class="string">"gpt-4o"</span>),
    ),
    tools=get_builtin_tools([<span class="string">"python_exec"</span>, <span class="string">"calculator"</span>]),
)

result = asyncio.run(agent.run(<span class="string">"What is the sum of the first 100 primes?"</span>))
<span class="function">print</span>(result)</code></pre>

<h2 id="qs-tools">2. Custom Tools</h2>
<p>Use the <code>@tool</code> decorator to turn any function into an agent tool.</p>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> tool

<span class="decorator">@tool</span>(name=<span class="string">"lookup_user"</span>, description=<span class="string">"Look up a user by ID"</span>)
<span class="keyword">def</span> <span class="function">lookup_user</span>(user_id: <span class="function">str</span>) -> <span class="function">str</span>:
    <span class="keyword">return</span> <span class="string">f"User {user_id}: John Doe, Engineering"</span></code></pre>

<h2 id="qs-guardrails">3. Add Guardrails</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> GuardrailChain
<span class="keyword">from</span> duxx_ai.governance.guardrails <span class="keyword">import</span> PIIGuardrail, PromptInjectionGuardrail

guardrails = GuardrailChain([
    PromptInjectionGuardrail(),
    PIIGuardrail(),
])

agent = Agent(config=config, guardrails=guardrails)</code></pre>

<h2 id="qs-graph">4. Graph Workflow</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Graph, EdgeCondition

graph = Graph(name=<span class="string">"support-workflow"</span>)
graph.add_node(<span class="string">"classify"</span>, handler=classify_ticket)
graph.add_node(<span class="string">"urgent"</span>, handler=handle_urgent)
graph.add_node(<span class="string">"normal"</span>, handler=handle_normal)

graph.set_entry_point(<span class="string">"classify"</span>)
graph.add_edge(<span class="string">"classify"</span>, <span class="string">"urgent"</span>,
    condition=EdgeCondition(key=<span class="string">"priority"</span>, value=<span class="string">"high"</span>))
graph.add_edge(<span class="string">"classify"</span>, <span class="string">"normal"</span>,
    condition=EdgeCondition(key=<span class="string">"priority"</span>, value=<span class="string">"low"</span>))

result = <span class="keyword">await</span> graph.run({<span class="string">"ticket"</span>: <span class="string">"Server is down!"</span>})</code></pre>

<h2 id="qs-next">Next Steps</h2>
<div class="feature-grid">
    <div class="feature-card" onclick="navigateTo('agents')">
        <h4>Core Concepts &#8594;</h4>
        <p>Deep dive into agents, tools, providers, and messages.</p>
    </div>
    <div class="feature-card" onclick="navigateTo('graph')">
        <h4>Graph Orchestration &#8594;</h4>
        <p>Build complex DAG workflows with conditional routing.</p>
    </div>
    <div class="feature-card" onclick="navigateTo('finetune')">
        <h4>Fine-Tuning &#8594;</h4>
        <p>Train domain-specific models from agent traces.</p>
    </div>
    <div class="feature-card" onclick="navigateTo('guardrails')">
        <h4>Governance &#8594;</h4>
        <p>Guardrails, RBAC, and audit logging for production.</p>
    </div>
</div>
</div>

<!-- ════════════════ AGENTS ════════════════ -->
<div id="page-agents" class="page">
<h1>Agents</h1>
<p class="lead">An Agent is the fundamental building block of Duxx AI. It combines an LLM, tools, guardrails, memory, and observability into a single autonomous unit.</p>

<h2 id="agent-config">AgentConfig</h2>
<p>Every agent is configured via an <code>AgentConfig</code> object:</p>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Agent, AgentConfig
<span class="keyword">from</span> duxx_ai.core.llm <span class="keyword">import</span> LLMConfig

config = AgentConfig(
    name=<span class="string">"my-agent"</span>,
    description=<span class="string">"A research assistant"</span>,
    system_prompt=<span class="string">"You are an expert researcher."</span>,
    llm=LLMConfig(provider=<span class="string">"openai"</span>, model=<span class="string">"gpt-4o"</span>),
    max_iterations=<span class="number">10</span>,
    max_tokens_per_turn=<span class="number">4096</span>,
)</code></pre>

<table>
    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>name</code></td><td><code>str</code></td><td><code>"agent"</code></td><td>Unique identifier for the agent</td></tr>
        <tr><td><code>description</code></td><td><code>str</code></td><td><code>""</code></td><td>Human-readable description</td></tr>
        <tr><td><code>system_prompt</code></td><td><code>str</code></td><td>Default prompt</td><td>System instructions for the LLM</td></tr>
        <tr><td><code>llm</code></td><td><code>LLMConfig</code></td><td>OpenAI GPT-4o</td><td>LLM provider configuration</td></tr>
        <tr><td><code>max_iterations</code></td><td><code>int</code></td><td><code>10</code></td><td>Max tool-calling loop iterations</td></tr>
        <tr><td><code>max_tokens_per_turn</code></td><td><code>int</code></td><td><code>4096</code></td><td>Token limit per LLM call</td></tr>
    </tbody>
</table>

<h2 id="agent-loop">The Agent Loop</h2>
<p>When you call <code>agent.run()</code>, the agent enters an iterative loop:</p>
<ol>
    <li><strong>Input guardrails</strong> check the user message</li>
    <li><strong>LLM call</strong> with conversation history and available tools</li>
    <li>If the LLM returns <strong>tool calls</strong>, execute them in parallel</li>
    <li>Append tool results and loop back to step 2</li>
    <li>If the LLM returns <strong>text only</strong>, run output guardrails and return</li>
</ol>

<div class="callout callout-info">
    <span class="callout-icon">&#9432;</span>
    <div><strong>Parallel Tool Execution:</strong> When the LLM requests multiple tool calls in a single turn, Duxx AI executes them concurrently via <code>asyncio.gather()</code> for maximum throughput.</div>
</div>

<h2 id="agent-tools">Registering Tools</h2>
<div class="code-header"><span>Python</span></div>
<pre><code>agent = Agent(config=config)
agent.register_tool(my_custom_tool)

<span class="comment"># Or pass tools at construction</span>
agent = Agent(config=config, tools=[tool_a, tool_b])</code></pre>

<h2 id="agent-state">Agent State</h2>
<p>Track runtime metrics via <code>agent.state</code>:</p>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="function">print</span>(agent.state.iteration)     <span class="comment"># Current iteration</span>
<span class="function">print</span>(agent.state.total_tokens)   <span class="comment"># Cumulative token usage</span>
<span class="function">print</span>(agent.state.total_cost)     <span class="comment"># Cumulative cost</span>
<span class="function">print</span>(agent.state.status)         <span class="comment"># idle | running | error</span></code></pre>
</div>

<!-- ════════════════ TOOLS ════════════════ -->
<div id="page-tools" class="page">
<h1>Tools</h1>
<p class="lead">Tools give agents the ability to take actions: execute code, read files, make HTTP requests, and more. Create custom tools from any Python function with the <code>@tool</code> decorator.</p>

<h2 id="tool-decorator">The @tool Decorator</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> tool

<span class="decorator">@tool</span>(
    name=<span class="string">"search_database"</span>,
    description=<span class="string">"Search the company database"</span>,
    requires_approval=<span class="keyword">True</span>,
    tags=[<span class="string">"database"</span>],
)
<span class="keyword">async def</span> <span class="function">search_database</span>(query: <span class="function">str</span>, limit: <span class="function">int</span> = <span class="number">10</span>) -> <span class="function">str</span>:
    <span class="comment"># Both sync and async functions are supported</span>
    results = <span class="keyword">await</span> db.search(query, limit=limit)
    <span class="keyword">return</span> json.dumps(results)</code></pre>

<p>The decorator automatically extracts parameter names, types, and defaults from the function signature to build the tool's JSON schema.</p>

<h2 id="builtin-tools">Built-in Tools</h2>
<table>
    <thead><tr><th>Name</th><th>Description</th><th>Approval</th></tr></thead>
    <tbody>
        <tr><td><code>python_exec</code></td><td>Execute Python code in a sandboxed subprocess</td><td>No</td></tr>
        <tr><td><code>bash_exec</code></td><td>Execute a shell command</td><td>Yes</td></tr>
        <tr><td><code>read_file</code></td><td>Read file contents</td><td>No</td></tr>
        <tr><td><code>write_file</code></td><td>Write content to a file</td><td>Yes</td></tr>
        <tr><td><code>list_files</code></td><td>List files in a directory</td><td>No</td></tr>
        <tr><td><code>web_request</code></td><td>Make HTTP requests (GET/POST/PUT/DELETE)</td><td>No</td></tr>
        <tr><td><code>json_query</code></td><td>Query JSON data with dot-notation paths</td><td>No</td></tr>
        <tr><td><code>calculator</code></td><td>Evaluate math expressions safely</td><td>No</td></tr>
    </tbody>
</table>

<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai.tools.builtin <span class="keyword">import</span> get_builtin_tools

<span class="comment"># Get specific tools</span>
tools = get_builtin_tools([<span class="string">"python_exec"</span>, <span class="string">"calculator"</span>])

<span class="comment"># Get all built-in tools</span>
all_tools = get_builtin_tools()</code></pre>

<h2 id="tool-execution">Tool Execution</h2>
<p>Tools support timeouts, retries, and both sync/async execution:</p>
<ul>
    <li><strong>Async functions</strong> are awaited directly</li>
    <li><strong>Sync functions</strong> run in a thread pool executor</li>
    <li><strong>Timeouts</strong> are enforced via <code>asyncio.wait_for()</code></li>
    <li><strong>Errors</strong> are caught and returned as <code>ToolResult</code> with error details</li>
</ul>
</div>

<!-- ════════════════ LLM PROVIDERS ════════════════ -->
<div id="page-providers" class="page">
<h1>LLM Providers</h1>
<p class="lead">Duxx AI supports multiple LLM providers through a unified interface. Switch between OpenAI, Anthropic, and local models with a single config change.</p>

<h2 id="provider-openai">OpenAI</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai.core.llm <span class="keyword">import</span> LLMConfig

config = LLMConfig(
    provider=<span class="string">"openai"</span>,
    model=<span class="string">"gpt-4o"</span>,
    api_key=<span class="string">"sk-..."</span>,
    temperature=<span class="number">0.7</span>,
    max_tokens=<span class="number">4096</span>,
)</code></pre>

<h2 id="provider-anthropic">Anthropic</h2>
<div class="code-header"><span>Python</span></div>
<pre><code>config = LLMConfig(
    provider=<span class="string">"anthropic"</span>,
    model=<span class="string">"claude-sonnet-4-6"</span>,
    api_key=<span class="string">"sk-ant-..."</span>,
)</code></pre>

<h2 id="provider-local">Local Models</h2>
<p>Use any OpenAI-compatible API (vLLM, Ollama, llama.cpp):</p>
<div class="code-header"><span>Python</span></div>
<pre><code>config = LLMConfig(
    provider=<span class="string">"local"</span>,
    model=<span class="string">"qwen2.5-7b"</span>,
    base_url=<span class="string">"http://localhost:8000/v1"</span>,
)</code></pre>

<div class="callout callout-tip">
    <span class="callout-icon">&#128161;</span>
    <div><strong>Adaptive Routing:</strong> Use the <a onclick="navigateTo('router')">AdaptiveRouter</a> to automatically select the best provider based on task complexity and cost.</div>
</div>
</div>

<!-- ════════════════ MESSAGES ════════════════ -->
<div id="page-messages" class="page">
<h1>Messages &amp; Conversations</h1>
<p class="lead">Duxx AI uses a structured message system for all agent communication, including tool calls and results.</p>

<h2 id="msg-roles">Message Roles</h2>
<table>
    <thead><tr><th>Role</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>Role.SYSTEM</code></td><td>System instructions</td></tr>
        <tr><td><code>Role.USER</code></td><td>User input</td></tr>
        <tr><td><code>Role.ASSISTANT</code></td><td>Agent / LLM response</td></tr>
        <tr><td><code>Role.TOOL</code></td><td>Tool execution results</td></tr>
        <tr><td><code>Role.ORCHESTRATOR</code></td><td>Orchestration control messages</td></tr>
    </tbody>
</table>

<h2 id="msg-conversation">Conversation</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Message, Role
<span class="keyword">from</span> duxx_ai.core.message <span class="keyword">import</span> Conversation

conv = Conversation()
conv.add(Message(role=Role.USER, content=<span class="string">"Hello!"</span>))

<span class="comment"># Access history</span>
last = conv.last_message
history = conv.get_history(last_n=<span class="number">5</span>)</code></pre>
</div>

<!-- ════════════════ GRAPH WORKFLOWS ════════════════ -->
<div id="page-graph" class="page">
<h1>Graph Workflows</h1>
<p class="lead">Build complex agent workflows as directed acyclic graphs (DAGs) with conditional routing, parallel execution, and state checkpointing. Inspired by LangGraph and state machine research.</p>

<div class="callout callout-note">
    <span class="callout-icon">&#128218;</span>
    <div><strong>Research Foundation:</strong> Duxx AI's graph engine draws from recent research on Graph of Thoughts (GoT), Tree of Thoughts (ToT), and state-machine-based agent control flow. See the <a href="#graph-research">Research Background</a> section below.</div>
</div>

<h2 id="graph-basic">Basic Graph</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Graph, EdgeCondition

graph = Graph(name=<span class="string">"my-workflow"</span>)

<span class="comment"># Define nodes (each node is an async handler)</span>
graph.add_node(<span class="string">"classify"</span>, handler=classify)
graph.add_node(<span class="string">"process"</span>, handler=process)
graph.add_node(<span class="string">"respond"</span>, handler=respond)

<span class="comment"># Wire the graph</span>
graph.set_entry_point(<span class="string">"classify"</span>)
graph.add_edge(<span class="string">"classify"</span>, <span class="string">"process"</span>)
graph.add_edge(<span class="string">"process"</span>, <span class="string">"respond"</span>)
graph.set_exit_point(<span class="string">"respond"</span>)

result = <span class="keyword">await</span> graph.run({<span class="string">"input"</span>: <span class="string">"Hello"</span>})</code></pre>

<h2 id="graph-conditional">Conditional Routing</h2>
<p>Route execution based on state values using <code>EdgeCondition</code>:</p>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="comment"># Supported operators: eq, neq, gt, lt, contains, exists</span>
graph.add_edge(<span class="string">"classify"</span>, <span class="string">"urgent"</span>,
    condition=EdgeCondition(key=<span class="string">"priority"</span>, value=<span class="string">"high"</span>, operator=<span class="string">"eq"</span>))
graph.add_edge(<span class="string">"classify"</span>, <span class="string">"normal"</span>,
    condition=EdgeCondition(key=<span class="string">"priority"</span>, value=<span class="string">"high"</span>, operator=<span class="string">"neq"</span>))</code></pre>

<h2 id="graph-parallel">Parallel Execution</h2>
<p>When multiple edges lead from a node, the graph engine detects and executes branches in parallel:</p>
<div class="code-header"><span>Python</span></div>
<pre><code>graph.add_edge(<span class="string">"start_node"</span>, <span class="string">"branch_a"</span>)
graph.add_edge(<span class="string">"start_node"</span>, <span class="string">"branch_b"</span>)
graph.add_edge(<span class="string">"start_node"</span>, <span class="string">"branch_c"</span>)
<span class="comment"># All three branches execute concurrently via asyncio.gather()</span></code></pre>

<h2 id="graph-state">GraphState</h2>
<p>State flows through the graph and is accessible at every node:</p>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">async def</span> <span class="function">my_handler</span>(state: GraphState) -> GraphState:
    data = state.get(<span class="string">"input"</span>)
    state.set(<span class="string">"result"</span>, process(data))
    <span class="keyword">return</span> state</code></pre>

<h2 id="graph-checkpoint">Checkpointing</h2>
<p>The graph automatically checkpoints state after each node execution, enabling recovery and debugging:</p>
<div class="code-header"><span>Python</span></div>
<pre><code>result = <span class="keyword">await</span> graph.run({<span class="string">"input"</span>: <span class="string">"data"</span>})

<span class="comment"># Access execution history</span>
<span class="keyword">for</span> checkpoint <span class="keyword">in</span> graph.checkpoints:
    <span class="function">print</span>(checkpoint.current_node, checkpoint.data)</code></pre>

<h2 id="graph-node-types">Node Types</h2>
<table>
    <thead><tr><th>Type</th><th>Purpose</th></tr></thead>
    <tbody>
        <tr><td><code>AGENT</code></td><td>Runs an AI agent with LLM</td></tr>
        <tr><td><code>TOOL</code></td><td>Executes a specific tool</td></tr>
        <tr><td><code>CONDITIONAL</code></td><td>Branching logic</td></tr>
        <tr><td><code>PARALLEL</code></td><td>Fan-out parallel execution</td></tr>
        <tr><td><code>HUMAN</code></td><td>Human-in-the-loop approval</td></tr>
        <tr><td><code>START</code> / <code>END</code></td><td>Graph entry and exit points</td></tr>
    </tbody>
</table>

<h2 id="graph-research">Research Background</h2>
<div class="callout callout-info">
    <span class="callout-icon">&#128214;</span>
    <div>
        <strong>Graph of Thoughts (Besta et al., 2023):</strong> Represents LLM reasoning as an arbitrary graph where thoughts can branch, merge, and loop &mdash; enabling more flexible problem solving than chain or tree structures.
        <br><br>
        <strong>Tree of Thoughts (Yao et al., 2023):</strong> Tree-based deliberation allowing LLMs to explore multiple reasoning paths with backtracking.
        <br><br>
        <strong>State Machines for Agents:</strong> Finite automaton approaches to agent control flow provide formal guarantees about execution paths, error handling, and termination conditions.
    </div>
</div>
</div>

<!-- ════════════════ CREW ════════════════ -->
<div id="page-crew" class="page">
<h1>Multi-Agent Crews</h1>
<p class="lead">Orchestrate multiple agents working together on complex tasks. Define roles, goals, and task dependencies for collaborative workflows.</p>

<h2 id="crew-basic">Creating a Crew</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Crew, CrewAgent
<span class="keyword">from</span> duxx_ai.orchestration.crew <span class="keyword">import</span> Task

crew = Crew(
    name=<span class="string">"content-team"</span>,
    agents=[
        CrewAgent(name=<span class="string">"researcher"</span>, role=<span class="string">"Researcher"</span>,
                  goal=<span class="string">"Find accurate information"</span>),
        CrewAgent(name=<span class="string">"writer"</span>, role=<span class="string">"Writer"</span>,
                  goal=<span class="string">"Write compelling content"</span>),
    ],
    tasks=[
        Task(id=<span class="string">"research"</span>, description=<span class="string">"Research AI trends"</span>,
             assigned_to=<span class="string">"researcher"</span>),
        Task(id=<span class="string">"write"</span>, description=<span class="string">"Write a blog post"</span>,
             assigned_to=<span class="string">"writer"</span>, dependencies=[<span class="string">"research"</span>]),
    ],
    strategy=<span class="string">"sequential"</span>,
)

result = <span class="keyword">await</span> crew.run()</code></pre>

<h2 id="crew-strategies">Execution Strategies</h2>
<table>
    <thead><tr><th>Strategy</th><th>Behavior</th></tr></thead>
    <tbody>
        <tr><td><code>sequential</code></td><td>Tasks run one-by-one, each receiving context from prior tasks</td></tr>
        <tr><td><code>parallel</code></td><td>Independent tasks run concurrently; respects dependency ordering</td></tr>
        <tr><td><code>hierarchical</code></td><td>A manager agent delegates and reviews work from other agents</td></tr>
    </tbody>
</table>

<h2 id="crew-deps">Task Dependencies</h2>
<p>Tasks with dependencies receive results from prerequisite tasks in their context automatically. The crew uses topological sorting to determine execution order.</p>
</div>

<!-- ════════════════ FINE-TUNE PIPELINE ════════════════ -->
<div id="page-finetune" class="page">
<h1>Fine-Tuning Pipeline</h1>
<p class="lead">Train domain-specific agent models from execution traces. Integrated Unsloth and HuggingFace PEFT support for efficient LoRA/QLoRA training.</p>

<span class="badge badge-enterprise">Enterprise</span>

<h2 id="ft-overview">How It Works</h2>
<ol>
    <li><strong>Collect traces</strong> &mdash; Agent execution traces are recorded via the observability system</li>
    <li><strong>Prepare dataset</strong> &mdash; Convert traces to chat or instruction format training data</li>
    <li><strong>Train</strong> &mdash; Fine-tune a base model with LoRA/QLoRA (Unsloth for 2x speed, 70% less VRAM)</li>
    <li><strong>Evaluate</strong> &mdash; Measure loss and task-specific metrics</li>
    <li><strong>Export</strong> &mdash; Save as SafeTensors or GGUF for deployment</li>
</ol>

<h2 id="ft-quick">Quick Example</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai.finetune.pipeline <span class="keyword">import</span> FineTunePipeline, TrainingConfig

pipeline = FineTunePipeline(
    training_config=TrainingConfig(
        base_model=<span class="string">"unsloth/Qwen2.5-7B"</span>,
        method=<span class="string">"lora"</span>,
        lora_r=<span class="number">16</span>,
        epochs=<span class="number">3</span>,
        load_in_4bit=<span class="keyword">True</span>,
    )
)

<span class="comment"># Convert agent traces to training data</span>
pipeline.prepare_dataset()

<span class="comment"># Train</span>
result = pipeline.train(<span class="string">"dataset.jsonl"</span>)
<span class="function">print</span>(<span class="string">f"Loss: {result.final_loss:.4f}"</span>)</code></pre>

<h2 id="ft-unsloth">Unsloth Integration</h2>
<div class="callout callout-tip">
    <span class="callout-icon">&#9889;</span>
    <div><strong>Performance:</strong> When Unsloth is installed, Duxx AI automatically uses it for 2x faster training and 70% less VRAM. Falls back to standard HuggingFace PEFT if unavailable.</div>
</div>
</div>

<!-- ════════════════ LORA ════════════════ -->
<div id="page-lora" class="page">
<h1>LoRA / QLoRA Training</h1>
<p class="lead">Parameter-efficient fine-tuning with Low-Rank Adaptation. Train 7B models on a single GPU.</p>

<h2 id="lora-config">Training Configuration</h2>
<table>
    <thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>base_model</code></td><td><code>"unsloth/Qwen2.5-7B"</code></td><td>HuggingFace model ID</td></tr>
        <tr><td><code>method</code></td><td><code>"lora"</code></td><td>Training method: lora, qlora, full</td></tr>
        <tr><td><code>lora_r</code></td><td><code>16</code></td><td>LoRA rank</td></tr>
        <tr><td><code>lora_alpha</code></td><td><code>32</code></td><td>LoRA scaling factor</td></tr>
        <tr><td><code>lora_dropout</code></td><td><code>0.05</code></td><td>Dropout rate</td></tr>
        <tr><td><code>epochs</code></td><td><code>3</code></td><td>Training epochs</td></tr>
        <tr><td><code>batch_size</code></td><td><code>4</code></td><td>Per-device batch size</td></tr>
        <tr><td><code>learning_rate</code></td><td><code>2e-4</code></td><td>Peak learning rate</td></tr>
        <tr><td><code>load_in_4bit</code></td><td><code>True</code></td><td>Enable 4-bit quantization (QLoRA)</td></tr>
        <tr><td><code>max_seq_length</code></td><td><code>2048</code></td><td>Maximum sequence length</td></tr>
    </tbody>
</table>

<h2 id="lora-export">Model Export</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="comment"># Export as SafeTensors (default)</span>
path = pipeline.export(format=<span class="string">"safetensors"</span>)

<span class="comment"># Export as GGUF for llama.cpp / Ollama</span>
path = pipeline.export(format=<span class="string">"gguf"</span>, quantize=<span class="string">"q4_k_m"</span>)</code></pre>
</div>

<!-- ════════════════ DATASET ════════════════ -->
<div id="page-dataset" class="page">
<h1>Dataset Preparation</h1>
<p class="lead">Convert agent execution traces into fine-tuning datasets. Supports chat and instruction formats.</p>

<h2 id="ds-trace">From Traces</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai.finetune.pipeline <span class="keyword">import</span> TraceToDataset

<span class="comment"># Convert trace JSONL to training dataset</span>
count = TraceToDataset.from_traces(
    <span class="string">"traces.jsonl"</span>,     <span class="comment"># Input: agent traces</span>
    <span class="string">"dataset.jsonl"</span>,    <span class="comment"># Output: training data</span>
    format=<span class="string">"chat"</span>,       <span class="comment"># "chat" or "instruction"</span>
)
<span class="function">print</span>(<span class="string">f"Generated {count} training samples"</span>)</code></pre>

<h2 id="ds-formats">Dataset Formats</h2>
<table>
    <thead><tr><th>Format</th><th>Structure</th><th>Use Case</th></tr></thead>
    <tbody>
        <tr><td><code>chat</code></td><td><code>{"messages": [...]}</code></td><td>Multi-turn conversations with tool calls</td></tr>
        <tr><td><code>instruction</code></td><td><code>{"instruction": ..., "output": ...}</code></td><td>Single-turn instruction following</td></tr>
    </tbody>
</table>
</div>

<!-- ════════════════ GUARDRAILS ════════════════ -->
<div id="page-guardrails" class="page">
<h1>Guardrails</h1>
<p class="lead">Protect your agents with input/output safety checks. Detect PII, block prompt injection, enforce token budgets, and filter content.</p>

<h2 id="gr-chain">Guardrail Chain</h2>
<p>Compose multiple guardrails into a chain that runs on every agent input and output:</p>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> GuardrailChain
<span class="keyword">from</span> duxx_ai.governance.guardrails <span class="keyword">import</span> (
    PIIGuardrail,
    PromptInjectionGuardrail,
    ContentFilterGuardrail,
    TokenBudgetGuardrail,
    TopicGuardrail,
)

chain = GuardrailChain([
    PromptInjectionGuardrail(),
    PIIGuardrail(allow_email=<span class="keyword">False</span>),
    ContentFilterGuardrail(blocked_patterns=[<span class="string">r"password"</span>, <span class="string">r"secret"</span>]),
    TokenBudgetGuardrail(max_tokens=<span class="number">100_000</span>),
    TopicGuardrail(blocked_topics=[<span class="string">"politics"</span>, <span class="string">"religion"</span>]),
])</code></pre>

<h2 id="gr-builtin">Built-in Guardrails</h2>
<table>
    <thead><tr><th>Guardrail</th><th>Direction</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>PromptInjectionGuardrail</code></td><td>Input</td><td>Detects common prompt injection patterns</td></tr>
        <tr><td><code>PIIGuardrail</code></td><td>Both</td><td>Blocks SSN, credit cards, phone numbers, emails</td></tr>
        <tr><td><code>ContentFilterGuardrail</code></td><td>Both</td><td>Regex-based content blocking</td></tr>
        <tr><td><code>TokenBudgetGuardrail</code></td><td>Both</td><td>Enforces cumulative token limits</td></tr>
        <tr><td><code>TopicGuardrail</code></td><td>Both</td><td>Restricts conversation topics</td></tr>
    </tbody>
</table>

<h2 id="gr-custom">Custom Guardrails</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Guardrail
<span class="keyword">from</span> duxx_ai.governance.guardrails <span class="keyword">import</span> GuardrailResult

<span class="keyword">class</span> <span class="function">MyGuardrail</span>(Guardrail):
    name = <span class="string">"my_guardrail"</span>

    <span class="keyword">async def</span> <span class="function">check</span>(self, text: <span class="function">str</span>, direction: <span class="function">str</span> = <span class="string">"input"</span>) -> GuardrailResult:
        <span class="keyword">if</span> <span class="string">"forbidden"</span> <span class="keyword">in</span> text.lower():
            <span class="keyword">return</span> GuardrailResult(passed=<span class="keyword">False</span>, reason=<span class="string">"Forbidden content"</span>)
        <span class="keyword">return</span> GuardrailResult(passed=<span class="keyword">True</span>)</code></pre>
</div>

<!-- ════════════════ RBAC ════════════════ -->
<div id="page-rbac" class="page">
<h1>Role-Based Access Control</h1>
<p class="lead">Enterprise RBAC with role inheritance. Control who can run agents, execute tools, and manage fine-tuning.</p>

<span class="badge badge-enterprise">Enterprise</span>

<h2 id="rbac-roles">Default Roles</h2>
<table>
    <thead><tr><th>Role</th><th>Inherits</th><th>Permissions</th></tr></thead>
    <tbody>
        <tr><td><strong>viewer</strong></td><td>&mdash;</td><td>Read agents and traces</td></tr>
        <tr><td><strong>operator</strong></td><td>viewer</td><td>+ Execute agents and tools</td></tr>
        <tr><td><strong>developer</strong></td><td>operator</td><td>+ Create/modify agents, fine-tuning</td></tr>
        <tr><td><strong>admin</strong></td><td>&mdash;</td><td>Full access to all resources</td></tr>
    </tbody>
</table>

<h2 id="rbac-usage">Usage</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai.governance.rbac <span class="keyword">import</span> RBACManager, User

rbac = RBACManager()
rbac.add_user(User(id=<span class="string">"user-1"</span>, name=<span class="string">"Alice"</span>))
rbac.assign_role(<span class="string">"user-1"</span>, <span class="string">"developer"</span>)

<span class="comment"># Check permissions</span>
rbac.check_permission(<span class="string">"user-1"</span>, <span class="string">"agent:researcher"</span>, <span class="string">"execute"</span>)  <span class="comment"># True</span>
rbac.check_permission(<span class="string">"user-1"</span>, <span class="string">"finetune:*"</span>, <span class="string">"write"</span>)        <span class="comment"># True</span></code></pre>
</div>

<!-- ════════════════ AUDIT ════════════════ -->
<div id="page-audit" class="page">
<h1>Audit Logging</h1>
<p class="lead">Append-only audit trail for compliance. Track every agent run, tool call, and guardrail trigger with risk levels.</p>

<span class="badge badge-enterprise">Enterprise</span>

<h2 id="audit-usage">Usage</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> AuditLog

audit = AuditLog(storage_path=<span class="string">"./audit.jsonl"</span>)

<span class="comment"># Log agent runs, tool calls, guardrail triggers</span>
audit.log_agent_run(<span class="string">"my-agent"</span>, user_input, output, user_id=<span class="string">"user-1"</span>)
audit.log_tool_call(<span class="string">"my-agent"</span>, <span class="string">"bash_exec"</span>, args, risk_level=<span class="string">"high"</span>)

<span class="comment"># Query audit log</span>
entries = audit.query(event_type=<span class="string">"tool_call"</span>, risk_level=<span class="string">"high"</span>)
summary = audit.get_risk_summary()</code></pre>
</div>

<!-- ════════════════ MEMORY ════════════════ -->
<div id="page-memory" class="page">
<h1>5-Tier Memory System</h1>
<p class="lead">A comprehensive memory architecture for agents: working, episodic, semantic, procedural, and shared memory.</p>

<h2 id="mem-tiers">Memory Tiers</h2>
<table>
    <thead><tr><th>Tier</th><th>Scope</th><th>Persistence</th><th>Use Case</th></tr></thead>
    <tbody>
        <tr><td><strong>Working</strong></td><td>Single task</td><td>TTL + LRU eviction</td><td>Current task context, scratch pad</td></tr>
        <tr><td><strong>Episodic</strong></td><td>Agent lifetime</td><td>File-backed JSONL</td><td>Past interactions, task history</td></tr>
        <tr><td><strong>Semantic</strong></td><td>Global</td><td>In-memory</td><td>Facts, domain knowledge</td></tr>
        <tr><td><strong>Procedural</strong></td><td>Global</td><td>In-memory</td><td>Learned workflows, SOPs</td></tr>
        <tr><td><strong>Shared</strong></td><td>Multi-agent</td><td>In-memory + locking</td><td>Cross-agent coordination</td></tr>
    </tbody>
</table>

<h2 id="mem-manager">MemoryManager</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> MemoryManager

memory = MemoryManager(storage_dir=<span class="string">"./memory"</span>)

<span class="comment"># Auto-route to the right tier</span>
memory.auto_store(<span class="string">"User prefers JSON format"</span>, context=<span class="string">"fact"</span>)
memory.auto_store(<span class="string">"Completed data migration"</span>, context=<span class="string">"result"</span>)

<span class="comment"># Search across all tiers</span>
results = memory.recall_all(<span class="string">"data migration"</span>, top_k=<span class="number">5</span>)</code></pre>

<h2 id="mem-shared">Shared Memory with Locking</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="comment"># Cross-agent coordination</span>
memory.shared.lock(<span class="string">"report_draft"</span>, agent_id=<span class="string">"writer"</span>)
memory.shared.write(<span class="string">"report_draft"</span>, content, agent_id=<span class="string">"writer"</span>)
memory.shared.unlock(<span class="string">"report_draft"</span>, agent_id=<span class="string">"writer"</span>)

<span class="comment"># Other agents can read</span>
draft = memory.shared.read(<span class="string">"report_draft"</span>)</code></pre>
</div>

<!-- ════════════════ ROUTER ════════════════ -->
<div id="page-router" class="page">
<h1>Adaptive Router</h1>
<p class="lead">Automatically route requests to the optimal model tier based on complexity, cost, and required capabilities. Enforce budget limits.</p>

<h2 id="router-setup">Setup</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> AdaptiveRouter
<span class="keyword">from</span> duxx_ai.router.adaptive <span class="keyword">import</span> ModelTier

router = AdaptiveRouter(
    tiers=[
        ModelTier(name=<span class="string">"small"</span>, config=local_config,
                  max_complexity=<span class="number">0.3</span>, cost_per_1k_input=<span class="number">0.0001</span>),
        ModelTier(name=<span class="string">"medium"</span>, config=openai_35_config,
                  max_complexity=<span class="number">0.6</span>, cost_per_1k_input=<span class="number">0.001</span>),
        ModelTier(name=<span class="string">"large"</span>, config=openai_4o_config,
                  max_complexity=<span class="number">1.0</span>, cost_per_1k_input=<span class="number">0.01</span>),
    ],
    budget_limit=<span class="number">50.0</span>,
)

response, decision = <span class="keyword">await</span> router.complete(<span class="string">"Explain quantum computing"</span>)
<span class="function">print</span>(<span class="string">f"Routed to: {decision.selected_tier}"</span>)
<span class="function">print</span>(<span class="string">f"Complexity: {decision.complexity_score:.2f}"</span>)</code></pre>

<h2 id="router-complexity">Complexity Estimation</h2>
<p>The router estimates task complexity using:</p>
<ul>
    <li><strong>Keyword analysis</strong> &mdash; detects complex vs. simple task indicators</li>
    <li><strong>Input length</strong> &mdash; longer prompts correlate with higher complexity</li>
    <li><strong>Conversation depth</strong> &mdash; multi-turn conversations score higher</li>
    <li><strong>Tool requirements</strong> &mdash; tasks requiring code or API calls score higher</li>
</ul>

<h2 id="router-stats">Routing Statistics</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="function">print</span>(router.stats.total_requests)
<span class="function">print</span>(router.stats.requests_per_tier)
<span class="function">print</span>(router.stats.total_cost)
<span class="function">print</span>(router.stats.avg_latency_per_tier)</code></pre>
</div>

<!-- ════════════════ TRACING ════════════════ -->
<div id="page-tracing" class="page">
<h1>Tracing &amp; Observability</h1>
<p class="lead">OpenTelemetry-compatible span-based tracing. Visualize agent execution, track costs, and export traces for fine-tuning.</p>

<h2 id="trace-basic">Basic Usage</h2>
<div class="code-header"><span>Python</span></div>
<pre><code><span class="keyword">from</span> duxx_ai <span class="keyword">import</span> Tracer
<span class="keyword">from</span> duxx_ai.observability.tracer <span class="keyword">import</span> ConsoleExporter, JSONExporter

tracer = Tracer(exporters=[
    ConsoleExporter(),                   <span class="comment"># Print to terminal</span>
    JSONExporter(<span class="string">"traces.jsonl"</span>),        <span class="comment"># Save for fine-tuning</span>
])

agent = Agent(config=config, tracer=tracer)</code></pre>

<h2 id="trace-spans">Span Hierarchy</h2>
<p>Duxx AI automatically creates nested spans:</p>
<div class="arch-diagram">agent.assistant.run (1200ms)
  &#9492; agent.assistant.llm_call (800ms)
  &#9492; tool.python_exec (350ms)
  &#9492; agent.assistant.llm_call (450ms)</div>

<h2 id="trace-cost">Cost Tracking</h2>
<div class="code-header"><span>Python</span></div>
<pre><code>summary = tracer.get_cost_summary()
<span class="function">print</span>(summary)
<span class="comment"># {"total_traces": 5, "total_spans": 23, "total_duration_ms": 4520.0}</span></code></pre>
</div>

<!-- ════════════════ API REFERENCE ════════════════ -->
<div id="page-api-reference" class="page">
<h1>API Reference</h1>
<p class="lead">Complete API documentation for all Duxx AI modules.</p>

<h2 id="api-core">duxx_ai.core</h2>
<h3>Agent</h3>
<table>
    <thead><tr><th>Method</th><th>Signature</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>run()</code></td><td><code>async (user_input: str, context: dict | None) -> str</code></td><td>Run the agent with user input</td></tr>
        <tr><td><code>register_tool()</code></td><td><code>(tool: Tool) -> None</code></td><td>Register a tool with the agent</td></tr>
        <tr><td><code>reset()</code></td><td><code>() -> None</code></td><td>Reset conversation and state</td></tr>
    </tbody>
</table>

<h3>Tool</h3>
<table>
    <thead><tr><th>Method</th><th>Signature</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>execute()</code></td><td><code>async (call: ToolCall) -> ToolResult</code></td><td>Execute the tool with arguments</td></tr>
        <tr><td><code>bind()</code></td><td><code>(fn: Callable) -> Tool</code></td><td>Bind a function to the tool</td></tr>
        <tr><td><code>to_schema()</code></td><td><code>() -> dict</code></td><td>Generate OpenAI-compatible tool schema</td></tr>
    </tbody>
</table>

<h2 id="api-orchestration">duxx_ai.orchestration</h2>
<h3>Graph</h3>
<table>
    <thead><tr><th>Method</th><th>Signature</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>add_node()</code></td><td><code>(node_id, handler, node_type) -> Graph</code></td><td>Add a node to the graph</td></tr>
        <tr><td><code>add_edge()</code></td><td><code>(source, target, condition) -> Graph</code></td><td>Add a directed edge</td></tr>
        <tr><td><code>set_entry_point()</code></td><td><code>(node_id) -> Graph</code></td><td>Set the graph entry point</td></tr>
        <tr><td><code>run()</code></td><td><code>async (initial_state) -> GraphState</code></td><td>Execute the graph</td></tr>
        <tr><td><code>visualize()</code></td><td><code>() -> str</code></td><td>ASCII visualization of the graph</td></tr>
    </tbody>
</table>

<h3>Crew</h3>
<table>
    <thead><tr><th>Method</th><th>Signature</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>add_agent()</code></td><td><code>(agent: CrewAgent) -> Crew</code></td><td>Add an agent to the crew</td></tr>
        <tr><td><code>add_task()</code></td><td><code>(task: Task) -> Crew</code></td><td>Add a task</td></tr>
        <tr><td><code>run()</code></td><td><code>async (input_data) -> CrewResult</code></td><td>Execute the crew workflow</td></tr>
    </tbody>
</table>

<h2 id="api-governance">duxx_ai.governance</h2>
<h3>GuardrailChain</h3>
<table>
    <thead><tr><th>Method</th><th>Signature</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>check_input()</code></td><td><code>async (text: str) -> GuardrailResult</code></td><td>Run all guardrails on input</td></tr>
        <tr><td><code>check_output()</code></td><td><code>async (text: str) -> GuardrailResult</code></td><td>Run all guardrails on output</td></tr>
        <tr><td><code>add()</code></td><td><code>(guardrail: Guardrail) -> GuardrailChain</code></td><td>Add a guardrail to the chain</td></tr>
    </tbody>
</table>

<h2 id="api-finetune">duxx_ai.finetune</h2>
<h3>FineTunePipeline</h3>
<table>
    <thead><tr><th>Method</th><th>Signature</th><th>Description</th></tr></thead>
    <tbody>
        <tr><td><code>prepare_dataset()</code></td><td><code>() -> str</code></td><td>Prepare training dataset from traces</td></tr>
        <tr><td><code>train()</code></td><td><code>(dataset_path: str | None) -> TrainingResult</code></td><td>Run fine-tuning</td></tr>
        <tr><td><code>evaluate()</code></td><td><code>(eval_config) -> EvalResult</code></td><td>Evaluate the trained model</td></tr>
        <tr><td><code>export()</code></td><td><code>(format, quantize) -> str</code></td><td>Export model (safetensors/gguf)</td></tr>
    </tbody>
</table>
</div>

<!-- ════════════════ CLI REFERENCE ════════════════ -->
<div id="page-cli-ref" class="page">
<h1>CLI Reference</h1>
<p class="lead">Command-line interface for managing agents, fine-tuning, and the Studio web UI.</p>

<h2 id="cli-commands">Commands</h2>

<h3><code>duxx_ai init</code></h3>
<p>Initialize a new Duxx AI project with a default <code>duxx_ai.json</code> configuration file.</p>
<div class="install-block"><span><span class="dollar">$</span> duxx_ai init</span></div>

<h3><code>duxx_ai run &lt;config&gt;</code></h3>
<p>Run an agent from a YAML or JSON configuration file.</p>
<div class="install-block"><span><span class="dollar">$</span> duxx_ai run duxx_ai.json -i "What is 2+2?"</span></div>

<h3><code>duxx_ai studio</code></h3>
<p>Launch the Duxx AI Studio web UI.</p>
<div class="install-block"><span><span class="dollar">$</span> duxx_ai studio --port 8080</span></div>

<h3><code>duxx_ai finetune prepare</code></h3>
<p>Convert agent traces to a training dataset.</p>
<div class="install-block"><span><span class="dollar">$</span> duxx_ai finetune prepare traces.jsonl -o dataset.jsonl</span></div>

<h3><code>duxx_ai finetune train</code></h3>
<p>Fine-tune a model on your dataset.</p>
<div class="install-block"><span><span class="dollar">$</span> duxx_ai finetune train --model unsloth/Qwen2.5-7B --dataset dataset.jsonl</span></div>

<h3><code>duxx_ai info</code></h3>
<p>Show system information and installed dependencies.</p>
<div class="install-block"><span><span class="dollar">$</span> duxx_ai info</span></div>
</div>

<!-- ══ Advanced Agent ══ -->
<div id="page-agent-advanced" class="page">
<h1>Advanced Agent Features</h1>
<p class="lead">Subagent spawning, planning, streaming, batch processing, and hooks.</p>

<h2 id="agent-stream">Streaming</h2>
<pre class="code-block"><code><span class="kw">async for</span> token <span class="kw">in</span> agent.stream(<span class="str">"Explain AI"</span>):
    print(token, end=<span class="str">""</span>, flush=<span class="val">True</span>)</code></pre>

<h2 id="agent-subagent">Subagent Spawning</h2>
<p>Spawn child agents with isolated context for subtask delegation.</p>
<pre class="code-block"><code>result = <span class="kw">await</span> agent.spawn_subagent(
    task=<span class="str">"Research quantum computing breakthroughs"</span>,
    name=<span class="str">"researcher-sub"</span>,
    system_prompt=<span class="str">"You are a physics researcher."</span>,
    isolated=<span class="val">True</span>,  <span class="cm"># Fresh conversation (no parent history)</span>
)</code></pre>

<h2 id="agent-plan">Plan and Execute</h2>
<p>Auto-decompose a complex objective into subtasks, run each via subagents, then synthesize.</p>
<pre class="code-block"><code>result = <span class="kw">await</span> agent.plan_and_execute(
    objective=<span class="str">"Build a complete market analysis for the EV industry"</span>,
    max_subtasks=<span class="val">5</span>,
)
print(result[<span class="str">"plan"</span>])          <span class="cm"># ["Research market size", ...]</span>
print(result[<span class="str">"final_answer"</span>])   <span class="cm"># Synthesized answer</span></code></pre>

<h2 id="agent-batch">Batch Processing</h2>
<pre class="code-block"><code>results = <span class="kw">await</span> agent.batch(
    [<span class="str">"What is AI?"</span>, <span class="str">"What is ML?"</span>, <span class="str">"What is DL?"</span>],
    max_concurrency=<span class="val">3</span>,
)</code></pre>

<h2 id="agent-retry">Retry + Fallback</h2>
<pre class="code-block"><code>agent = Agent(config=AgentConfig(
    llm=LLMConfig(provider=<span class="str">"openai"</span>, model=<span class="str">"gpt-4o"</span>),
    fallback_llm=LLMConfig(provider=<span class="str">"anthropic"</span>, model=<span class="str">"claude-sonnet-4-20250514"</span>),
    retry=RetryConfig(max_retries=<span class="val">3</span>, backoff_factor=<span class="val">1.5</span>),
))</code></pre>

<h2 id="agent-hooks">Pre/Post Model Hooks</h2>
<pre class="code-block"><code><span class="kw">def</span> <span class="fn">log_before</span>(conversation):
    print(<span class="str">f"Calling LLM with {len(conversation.messages)} msgs"</span>)
    <span class="kw">return</span> conversation

agent.add_pre_hook(log_before)
agent.add_post_hook(<span class="kw">lambda</span> resp: resp)  <span class="cm"># Inspect/modify response</span></code></pre>
</div>

<!-- ══ Output Parsers ══ -->
<div id="page-parsers" class="page">
<h1>Output Parsers</h1>
<p class="lead">6 built-in parsers to extract structured data from LLM responses.</p>

<h2 id="parser-json">JSONOutputParser</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.parsers <span class="kw">import</span> JSONOutputParser
parser = JSONOutputParser()
data = parser.parse(<span class="str">'```json\n{"score": 95}\n```'</span>)  <span class="cm"># Handles code fences</span></code></pre>

<h2 id="parser-pydantic">PydanticOutputParser</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.parsers <span class="kw">import</span> PydanticOutputParser
<span class="kw">class</span> <span class="fn">Analysis</span>(BaseModel):
    sentiment: str
    confidence: float

parser = PydanticOutputParser(Analysis)
print(parser.get_format_instructions())  <span class="cm"># Inject into prompt</span>
result = parser.parse(<span class="str">'{"sentiment": "positive", "confidence": 0.9}'</span>)</code></pre>

<h2 id="parser-markdown">MarkdownOutputParser</h2>
<pre class="code-block"><code>parser = MarkdownOutputParser(sections=[<span class="str">"Summary"</span>, <span class="str">"Actions"</span>])
result = parser.parse(<span class="str">"## Summary\nGood.\n## Actions\n- Deploy"</span>)</code></pre>

<h2 id="parser-list">ListOutputParser</h2>
<pre class="code-block"><code>items = ListOutputParser().parse(<span class="str">"1. First\n2. Second\n- Third"</span>)</code></pre>

<h2 id="parser-regex">RegexOutputParser</h2>
<pre class="code-block"><code>parser = RegexOutputParser(<span class="str">r"Score: (?P&lt;score&gt;\\d+)/100"</span>)
result = parser.parse(<span class="str">"Score: 85/100"</span>)  <span class="cm"># {"score": "85"}</span></code></pre>

<h2 id="parser-structured">Structured Output from LLM</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> with_structured_output_async
result = <span class="kw">await</span> with_structured_output_async(provider, Analysis, conversation)</code></pre>
</div>

<!-- ══ LLM Cache & Rate Limit ══ -->
<div id="page-llm-advanced" class="page">
<h1>LLM Cache &amp; Rate Limiting</h1>
<p class="lead">Response caching, token-bucket rate limiting, and the CachedProvider wrapper.</p>

<h2 id="llm-cache">LLMCache</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> LLMCache
cache = LLMCache(ttl_seconds=<span class="val">300</span>, max_entries=<span class="val">1000</span>)
cached = cache.get(conversation, tools, system_prompt)
<span class="kw">if</span> cached: <span class="kw">return</span> cached
cache.set(conversation, tools, system_prompt, response)
print(cache.stats)  <span class="cm"># {"hits": 5, "misses": 2, "entries": 7}</span></code></pre>

<h2 id="llm-rate">RateLimiter</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> RateLimiter
limiter = RateLimiter(max_requests_per_minute=<span class="val">60</span>)
<span class="kw">await</span> limiter.acquire()  <span class="cm"># Blocks if rate exceeded</span></code></pre>

<h2 id="llm-cached-provider">CachedProvider (all-in-one)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> CachedProvider, create_provider
base = create_provider(LLMConfig(model=<span class="str">"gpt-4o"</span>))
provider = CachedProvider(base, cache_ttl=<span class="val">300</span>, rate_limit_rpm=<span class="val">60</span>)</code></pre>
</div>

<!-- ══ Advanced Graph ══ -->
<div id="page-graph-advanced" class="page">
<h1>Advanced Graph Features</h1>
<p class="lead">Subgraphs, streaming modes, time-travel debugging, node caching, and deferred nodes.</p>

<h2 id="graph-subgraph">Subgraph Composition</h2>
<pre class="code-block"><code>inner = Graph(<span class="str">"data-pipeline"</span>)
<span class="cm"># ... build inner graph ...</span>
outer = Graph(<span class="str">"main"</span>)
outer.add_subgraph(<span class="str">"data"</span>, inner,
    input_mapping={<span class="str">"raw_url"</span>: <span class="str">"source"</span>},
    output_mapping={<span class="str">"cleaned"</span>: <span class="str">"data"</span>},
    isolated=<span class="val">True</span>,
)</code></pre>

<h2 id="graph-stream">Streaming Modes</h2>
<pre class="code-block"><code><span class="cm"># values — full state after each node</span>
<span class="kw">async for</span> event <span class="kw">in</span> graph.stream(data, mode=<span class="str">"values"</span>): ...

<span class="cm"># updates — only changed keys per node</span>
<span class="kw">async for</span> event <span class="kw">in</span> graph.stream(data, mode=<span class="str">"updates"</span>): ...

<span class="cm"># debug — full execution info (node, duration, state diff)</span>
<span class="kw">async for</span> event <span class="kw">in</span> graph.stream(data, mode=<span class="str">"debug"</span>):
    print(event[<span class="str">"node"</span>], event[<span class="str">"duration_ms"</span>], event[<span class="str">"changes"</span>])</code></pre>

<h2 id="graph-timetravel">Time-Travel Debugging</h2>
<pre class="code-block"><code><span class="cm"># List all checkpoints</span>
checkpoints = graph.list_checkpoints()

<span class="cm"># Replay from any checkpoint</span>
states = <span class="kw">await</span> graph.replay(from_checkpoint=<span class="val">0</span>)

<span class="cm"># Fork with modified state (explore alternatives)</span>
forked = <span class="kw">await</span> graph.fork(checkpoint_index=<span class="val">2</span>, state_overrides={<span class="str">"approved"</span>: <span class="val">False</span>})

<span class="cm"># Persist checkpoints to disk</span>
graph.save_checkpoints(<span class="str">"checkpoints.json"</span>)
graph.load_checkpoints(<span class="str">"checkpoints.json"</span>)</code></pre>

<h2 id="graph-cache">Node Caching</h2>
<pre class="code-block"><code>graph.enable_node_caching()  <span class="cm"># Skip redundant node execution</span>
graph.clear_cache()</code></pre>

<h2 id="graph-deferred">Deferred Nodes</h2>
<pre class="code-block"><code>graph.add_deferred_node(<span class="str">"final"</span>, handler, wait_for=[<span class="str">"step_a"</span>, <span class="str">"step_b"</span>])</code></pre>
</div>

<!-- ══ RAG ══ -->
<div id="page-rag" class="page">
<h1>RAG Pipeline</h1>
<p class="lead">Retrieval-Augmented Generation: load, split, embed, store, and retrieve documents.</p>

<h2 id="rag-load">Document Loaders (5 types)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag <span class="kw">import</span> TextLoader, CSVLoader, JSONLLoader, WebLoader
docs = TextLoader(<span class="str">"knowledge.txt"</span>).load()
docs = CSVLoader(<span class="str">"data.csv"</span>, content_columns=[<span class="str">"q"</span>, <span class="str">"a"</span>]).load()
docs = JSONLLoader(<span class="str">"data.jsonl"</span>).load()
docs = WebLoader(<span class="str">"https://example.com"</span>).load()</code></pre>

<h2 id="rag-split">Text Splitters (3 types)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag <span class="kw">import</span> CharacterSplitter, RecursiveSplitter, TokenSplitter
chunks = RecursiveSplitter(chunk_size=<span class="val">500</span>, chunk_overlap=<span class="val">50</span>).split_many(docs)</code></pre>

<h2 id="rag-embed">Embed + Store + Search</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag <span class="kw">import</span> LocalEmbedder, InMemoryVectorStore
embedder = LocalEmbedder(dimension=<span class="val">384</span>)
store = InMemoryVectorStore(embedder)
store.add(chunks)
results = store.search(<span class="str">"fine-tuning"</span>, top_k=<span class="val">5</span>)</code></pre>

<h2 id="rag-retrieve">Retrievers (3 types)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag <span class="kw">import</span> VectorRetriever, KeywordRetriever, HybridRetriever
hybrid = HybridRetriever(
    VectorRetriever(store), KeywordRetriever(chunks),
    vector_weight=<span class="val">0.7</span>, keyword_weight=<span class="val">0.3</span>,
)
docs = hybrid.retrieve(<span class="str">"LoRA parameters"</span>, top_k=<span class="val">5</span>)</code></pre>
</div>

<!-- ══ Data Connectors ══ -->
<div id="page-connectors" class="page">
<h1>Data Connectors</h1>
<p class="lead">Load training data from 7 sources: HuggingFace, S3, GCS, Azure, Google Drive, Database, Local.</p>

<h2 id="conn-hf">HuggingFace</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.datasets <span class="kw">import</span> DatasetManager
mgr = DatasetManager()
mgr.load_from_huggingface(<span class="str">"tatsu-lab/alpaca"</span>, max_samples=<span class="val">5000</span>)</code></pre>

<h2 id="conn-s3">AWS S3</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.connectors <span class="kw">import</span> S3Connector
s3 = S3Connector(bucket=<span class="str">"my-data"</span>, key=<span class="str">"train.csv"</span>)
mgr.load_from_connector(s3)</code></pre>

<h2 id="conn-gcs">Google Cloud Storage</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.connectors <span class="kw">import</span> GCSConnector
gcs = GCSConnector(bucket=<span class="str">"my-bucket"</span>, blob_name=<span class="str">"data.jsonl"</span>)
mgr.load_from_connector(gcs)</code></pre>

<h2 id="conn-azure">Azure Blob</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.connectors <span class="kw">import</span> AzureBlobConnector
azure = AzureBlobConnector(container=<span class="str">"data"</span>, blob_name=<span class="str">"train.parquet"</span>)
mgr.load_from_connector(azure)</code></pre>

<h2 id="conn-db">Database (SQL + MongoDB)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.connectors <span class="kw">import</span> DatabaseConnector
db = DatabaseConnector(connection_string=<span class="str">"postgresql://..."</span>, query=<span class="str">"SELECT * FROM data"</span>)
mgr.load_from_connector(db)</code></pre>

<h2 id="conn-convert">Auto-Convert Raw Formats</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.converter <span class="kw">import</span> FormatConverter
raw = [{<span class="str">"question"</span>: <span class="str">"What is AI?"</span>, <span class="str">"answer"</span>: <span class="str">"..."</span>}]
training = FormatConverter.auto_convert(raw, target_format=<span class="str">"chat"</span>)</code></pre>

<h2 id="conn-popular">Popular HuggingFace Datasets (26 curated)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.popular_datasets <span class="kw">import</span> list_popular_datasets
datasets = list_popular_datasets(<span class="str">"Chat"</span>)  <span class="cm"># Filter by category</span></code></pre>
</div>

<!-- ══ Tinker Cloud ══ -->
<div id="page-tinker" class="page">
<h1>Tinker Cloud Fine-Tuning</h1>
<p class="lead">Train on 40+ models (up to 397B params) with pay-per-token pricing. No local GPU required.</p>

<h2 id="tinker-setup">Setup</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.tinker <span class="kw">import</span> TinkerProvider, TinkerConfig
provider = TinkerProvider(TinkerConfig(
    api_key=<span class="str">"tk-..."</span>,
    base_model=<span class="str">"meta-llama/Llama-3.1-8B"</span>,
    lora_rank=<span class="val">32</span>,
    learning_rate=<span class="val">1e-4</span>,
))</code></pre>

<h2 id="tinker-cost">Estimate Cost</h2>
<pre class="code-block"><code>cost = provider.estimate_cost(<span class="str">"llama-3.1-8b"</span>, dataset_tokens=<span class="val">500000</span>, epochs=<span class="val">3</span>)
print(<span class="str">f"Estimated: ${cost['total_estimated_usd']}"</span>)</code></pre>

<h2 id="tinker-train">Train</h2>
<pre class="code-block"><code>job = provider.create_job(<span class="str">"my-finetune"</span>, dataset_path=<span class="str">"train.jsonl"</span>)
result = <span class="kw">await</span> provider.start_training(job.id)</code></pre>

<h2 id="tinker-models">22 Supported Models</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.tinker <span class="kw">import</span> list_tinker_models
<span class="kw">for</span> m <span class="kw">in</span> list_tinker_models():
    print(<span class="str">f"{m['name']:30s} ${m['price_train']:.2f}/1M tok"</span>)</code></pre>
</div>

<!-- ══ Model Storage ══ -->
<div id="page-model-storage" class="page">
<h1>Model Storage</h1>
<p class="lead">Export trained models to S3, GCS, Azure, HuggingFace Hub, or Duxx AI Cloud.</p>

<h2 id="storage-s3">AWS S3</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.storage <span class="kw">import</span> S3ModelStorage
s3 = S3ModelStorage(bucket=<span class="str">"my-models"</span>)
uri = s3.upload(<span class="str">"./duxx_ai-finetuned"</span>, <span class="str">"my-model-v1"</span>)
<span class="cm"># s3://my-models/duxx_ai-models/my-model-v1</span></code></pre>

<h2 id="storage-hf">HuggingFace Hub</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.storage <span class="kw">import</span> HuggingFaceHubStorage
hf = HuggingFaceHubStorage(token=<span class="str">"hf_..."</span>)
uri = hf.upload(<span class="str">"./model"</span>, <span class="str">"my-org/my-model"</span>)</code></pre>

<h2 id="storage-cloud">Duxx AI Cloud (Managed)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.finetune.storage <span class="kw">import</span> DuxxCloudStorage
cloud = DuxxCloudStorage(api_key=<span class="str">"dxx-..."</span>)
uri = cloud.upload(<span class="str">"./model"</span>, <span class="str">"my-model-v1"</span>)
<span class="cm"># duxx_ai://my-org/my-model-v1</span></code></pre>
</div>

<!-- ══ Evaluation ══ -->
<div id="page-evaluation" class="page">
<h1>Agent Evaluation</h1>
<p class="lead">12 scoring metrics, A/B testing, regression detection, human annotations, experiment tracking.</p>

<h2 id="eval-basic">Multi-Metric Evaluation</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.observability.evaluator <span class="kw">import</span> AgentEvaluator, EvalCase
evaluator = AgentEvaluator(
    scorers=[<span class="str">"contains"</span>, <span class="str">"coherence"</span>, <span class="str">"safety"</span>, <span class="str">"faithfulness"</span>],
    pass_threshold=<span class="val">0.6</span>,
)
cases = [EvalCase(input=<span class="str">"What is AI?"</span>, expected_output=<span class="str">"artificial intelligence"</span>)]
result = <span class="kw">await</span> evaluator.evaluate(agent, cases, name=<span class="str">"v1.0"</span>)
print(result.summary())
print(result.metric_averages)     <span class="cm"># Per-scorer breakdown</span>
print(result.scores_by_category)  <span class="cm"># By category</span></code></pre>

<h2 id="eval-scorers">12 Built-in Scorers</h2>
<p><code>exact_match</code>, <code>contains</code>, <code>keyword</code>, <code>tool_call</code>, <code>semantic_similarity</code>, <code>length_ratio</code>, <code>format_compliance</code>, <code>coherence</code>, <code>safety</code>, <code>faithfulness</code>, <code>latency</code>, <code>cost</code></p>

<h2 id="eval-ab">A/B Testing (Pairwise)</h2>
<pre class="code-block"><code>results = <span class="kw">await</span> evaluator.pairwise_compare(agent_v1, agent_v2, cases)
wins_a = sum(<span class="val">1</span> <span class="kw">for</span> r <span class="kw">in</span> results <span class="kw">if</span> r.winner == <span class="str">"A"</span>)</code></pre>

<h2 id="eval-regression">Regression Detection</h2>
<pre class="code-block"><code>regression = evaluator.detect_regression(new_result, baseline)
<span class="kw">if</span> regression.regressed > <span class="val">0</span>:
    print(<span class="str">f"Regressed: {regression.regressed_cases}"</span>)</code></pre>

<h2 id="eval-human">Human Annotations</h2>
<pre class="code-block"><code>evaluator.add_annotation(<span class="str">"case-1"</span>, annotator=<span class="str">"alice"</span>, score=<span class="val">0.9</span>, label=<span class="str">"good"</span>)
print(evaluator.annotation_agreement())</code></pre>

<h2 id="eval-experiments">Experiment Tracking</h2>
<pre class="code-block"><code>experiments = evaluator.list_experiments()
comparison = evaluator.compare_experiments(exp_a_id, exp_b_id)</code></pre>
</div>

<!-- ══ Templates ══ -->
<div id="page-templates" class="page">
<h1>Enterprise Agent Templates</h1>
<p class="lead">14 production-ready agents with domain-specific tools, guardrails, and system prompts.</p>

<h2 id="tmpl-list">Available Templates</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.templates <span class="kw">import</span> TEMPLATES
<span class="kw">for</span> name, cls <span class="kw">in</span> TEMPLATES.items():
    info = cls.info()
    print(<span class="str">f"  {name:25s} [{info['category']}]"</span>)</code></pre>
<p><strong>Templates:</strong> EmailAgent, CallCenterAgent, MarketingAgent, InvestmentBankerAgent, PortfolioManagerAgent, DeepResearcherAgent, CodeBuilderAgent, SecurityAgent, DevOpsAgent, ComplianceAgent, FinanceManagerAgent, VirtualCFO, VirtualCMO, VirtualCHRO</p>

<h2 id="tmpl-use">Create and Use</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.templates <span class="kw">import</span> DeepResearcherAgent
researcher = DeepResearcherAgent.create()
result = <span class="kw">await</span> researcher.run(<span class="str">"Analyze AI market trends"</span>)</code></pre>
</div>

<!-- ══ REST API ══ -->
<div id="page-rest-api" class="page">
<h1>REST API Reference (43 Endpoints)</h1>
<p class="lead">Programmatic access to all Duxx AI features via HTTP.</p>

<h2 id="rest-chat">Chat &amp; Agents</h2>
<pre class="code-block"><code>POST /api/chat              <span class="cm"># Send message to agent</span>
GET  /api/tools             <span class="cm"># List all tools</span>
GET  /api/agents            <span class="cm"># List agents</span>
GET  /api/templates         <span class="cm"># List 14 templates</span></code></pre>

<h2 id="rest-finetune">Fine-Tuning</h2>
<pre class="code-block"><code>GET  /api/finetune/models        <span class="cm"># 12 local models</span>
POST /api/finetune/jobs          <span class="cm"># Create training job</span>
GET  /api/finetune/jobs          <span class="cm"># List jobs</span>
GET  /api/finetune/quant-methods <span class="cm"># 14 GGUF methods</span></code></pre>

<h2 id="rest-datasets">Datasets</h2>
<pre class="code-block"><code>GET  /api/datasets/popular       <span class="cm"># 26 HuggingFace datasets</span>
POST /api/datasets/load-hf      <span class="cm"># Load from HuggingFace</span>
POST /api/datasets/convert      <span class="cm"># Convert format</span>
POST /api/datasets/connect      <span class="cm"># Test data source</span>
POST /api/finetune/datasets      <span class="cm"># Create dataset</span>
GET  /api/finetune/datasets/{id}/preview
GET  /api/finetune/datasets/{id}/stats
POST /api/finetune/datasets/{id}/validate
POST /api/finetune/datasets/{id}/split</code></pre>

<h2 id="rest-tinker">Tinker Cloud</h2>
<pre class="code-block"><code>GET  /api/tinker/models          <span class="cm"># 22 cloud models</span>
POST /api/tinker/jobs            <span class="cm"># Create cloud job</span>
POST /api/tinker/estimate-cost   <span class="cm"># Cost calculator</span></code></pre>

<h2 id="rest-export">Model Export</h2>
<pre class="code-block"><code>POST /api/models/export          <span class="cm"># Upload to S3/GCS/Azure/HF</span>
GET  /api/models/storage         <span class="cm"># List stored models</span></code></pre>

<h2 id="rest-governance">Governance &amp; Health</h2>
<pre class="code-block"><code>GET  /api/governance             <span class="cm"># Guardrails + RBAC info</span>
GET  /api/health                 <span class="cm"># {"status": "ok"}</span></code></pre>
</div>

<!-- ══════ AGENT PATTERNS ══════ -->

<div id="page-react" class="page">
<h1>ReAct Agent</h1>
<p>Reasoning + Acting loop with self-correction and reflection. Think → Act → Observe → Verify.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> ReActAgent
<span class="kw">from</span> duxx_ai.core.agent <span class="kw">import</span> Agent, AgentConfig

base = Agent(config=AgentConfig(name=<span class="string">"researcher"</span>), tools=[search, calculator])
react = ReActAgent(base, max_iterations=<span class="number">10</span>, reflection_threshold=<span class="number">3</span>, verify_answer=<span class="number">True</span>)
trace = <span class="kw">await</span> react.run(<span class="string">"What is the GDP of France divided by its population?"</span>)

<span class="kw">print</span>(trace.final_answer)
<span class="kw">print</span>(f<span class="string">"Steps: {len(trace.steps)}, Corrections: {trace.self_corrections}"</span>)
<span class="kw">print</span>(f<span class="string">"Reflection triggered: {trace.reflection_triggered}"</span>)</code></pre>
<h2>Parameters</h2>
<ul>
<li><code>max_iterations</code> — Maximum think-act-observe cycles (default: 10)</li>
<li><code>reflection_threshold</code> — Steps without progress before triggering reflection (default: 3)</li>
<li><code>verify_answer</code> — Self-verify the final answer before returning (default: True)</li>
</ul>
</div>

<div id="page-handoffs" class="page">
<h1>Agent Handoffs</h1>
<p>Tool-based agent-to-agent delegation. One agent transfers control to a specialist mid-conversation (like OpenAI Agents SDK).</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> AgentHandoff

handoff = AgentHandoff()
handoff.register(<span class="string">"refund_agent"</span>, refund_agent, <span class="string">"Handles refund requests"</span>)
handoff.register(<span class="string">"billing_agent"</span>, billing_agent, <span class="string">"Handles billing questions"</span>)

<span class="cm"># Triage agent gets handoff tools automatically</span>
triage = Agent(config=AgentConfig(name=<span class="string">"triage"</span>), tools=handoff.as_tools())
result = <span class="kw">await</span> triage.run(<span class="string">"I want a refund for my last order"</span>)
<span class="cm"># triage calls transfer_to_refund_agent → refund_agent handles it</span>

<span class="cm"># View handoff history</span>
<span class="kw">for</span> h <span class="kw">in</span> handoff.history:
    <span class="kw">print</span>(f<span class="string">"{h.source_agent} → {h.target_agent}: {h.task}"</span>)</code></pre>
</div>

<div id="page-self-improving" class="page">
<h1>Self-Improving Agent</h1>
<p>Agents that learn from rewards and accumulate skills over time (DeepAgents SAGE pattern).</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> SelfImprovingAgent

sia = SelfImprovingAgent(base_agent, skill_persistence_path=<span class="string">"skills.json"</span>)

<span class="cm"># Run with automatic self-evaluation</span>
result = <span class="kw">await</span> sia.run(<span class="string">"Analyze this financial report"</span>)

<span class="cm"># Provide human feedback (optional)</span>
sia.reward(query=<span class="string">"Analyze..."</span>, response=result, score=<span class="number">0.9</span>, feedback=<span class="string">"Great analysis"</span>)

<span class="cm"># Check improvement stats</span>
<span class="kw">print</span>(sia.stats)
<span class="cm"># {'total_runs': 10, 'avg_score': 0.82, 'improvement': +0.15, 'skills_learned': 3}</span></code></pre>
</div>

<div id="page-teachable" class="page">
<h1>Teachable Agent</h1>
<p>Agent that learns facts, preferences, and corrections from conversations (like AutoGen teachable agents).</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> TeachableAgent

ta = TeachableAgent(base_agent, memory_path=<span class="string">"memory.json"</span>)

<span class="cm"># Agent learns from conversations</span>
<span class="kw">await</span> ta.run(<span class="string">"My name is Bankatesh and I prefer formal reports"</span>)

<span class="cm"># Agent remembers and applies knowledge</span>
<span class="kw">await</span> ta.run(<span class="string">"Write me a report"</span>)  <span class="cm"># Uses formal style, addresses as Bankatesh</span>

<span class="cm"># Explicit teaching</span>
ta.teach(<span class="string">"preferred_language"</span>, <span class="string">"Python"</span>)
ta.correct(<span class="string">"report_format"</span>, <span class="string">"Use bullet points, not paragraphs"</span>)

<span class="cm"># View learned knowledge</span>
<span class="kw">print</span>(ta.knowledge)  <span class="cm"># {'user_name': 'Bankatesh', 'report_style': 'formal', ...}</span></code></pre>
</div>

<div id="page-eval-optimizer" class="page">
<h1>Evaluator-Optimizer</h1>
<p>Two-agent generate-evaluate-revise loop until quality threshold is met (Anthropic pattern).</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> EvaluatorOptimizer

eo = EvaluatorOptimizer(
    generator=writer_agent,
    evaluator=critic_agent,  <span class="cm"># or None for self-evaluation</span>
    threshold=<span class="number">0.85</span>,
    max_iterations=<span class="number">3</span>,
)
result = <span class="kw">await</span> eo.run(<span class="string">"Write a technical blog post about RAG pipelines"</span>)

<span class="kw">print</span>(f<span class="string">"Final score: {result.final_score}"</span>)
<span class="kw">print</span>(f<span class="string">"Iterations: {result.iterations}"</span>)
<span class="kw">print</span>(f<span class="string">"Improvements applied: {result.improvements}"</span>)</code></pre>
</div>

<div id="page-orchestrator-worker" class="page">
<h1>Orchestrator-Worker</h1>
<p>Dynamic task decomposition with specialist worker delegation.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> OrchestratorWorker

ow = OrchestratorWorker(orchestrator=manager_agent)
ow.add_worker(<span class="string">"researcher"</span>, research_agent, <span class="string">"Deep research and analysis"</span>)
ow.add_worker(<span class="string">"writer"</span>, writer_agent, <span class="string">"Writing and editing"</span>)
ow.add_worker(<span class="string">"coder"</span>, code_agent, <span class="string">"Code generation"</span>)

result = <span class="kw">await</span> ow.run(<span class="string">"Build a Python library for sentiment analysis with docs"</span>)
<span class="cm"># Orchestrator decomposes → assigns to coder + writer → synthesizes</span></code></pre>
</div>

<div id="page-parallel-guardrails" class="page">
<h1>Parallel Guardrails</h1>
<p>Run multiple guardrails in parallel with blocking/non-blocking/passive modes (like OpenAI Agents SDK).</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> ParallelGuardrails, GuardrailMode

pg = ParallelGuardrails()
pg.add(<span class="string">"pii_check"</span>, pii_filter_fn, mode=GuardrailMode.BLOCKING)
pg.add(<span class="string">"toxicity"</span>, toxicity_fn, mode=GuardrailMode.NON_BLOCKING)
pg.add(<span class="string">"cost_log"</span>, cost_logger_fn, mode=GuardrailMode.PASSIVE)

<span class="cm"># Check input — BLOCKING runs first, NON_BLOCKING in parallel</span>
results = <span class="kw">await</span> pg.check_input(<span class="string">"user message"</span>)
<span class="kw">if</span> pg.any_blocked(results):
    <span class="kw">return</span> <span class="string">"Input blocked by guardrail"</span>

<span class="cm"># Check output after LLM call</span>
results = <span class="kw">await</span> pg.check_output(<span class="string">"llm response"</span>)
<span class="kw">print</span>(pg.summary(results))</code></pre>
</div>

<div id="page-agentic-rag" class="page">
<h1>Agentic RAG</h1>
<p>Agent-driven query rewriting with multi-hop retrieval and source verification.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.patterns <span class="kw">import</span> AgenticRAG

rag = AgenticRAG(agent=researcher, retriever=vector_store, max_hops=<span class="number">2</span>, top_k=<span class="number">5</span>)
result = <span class="kw">await</span> rag.query(<span class="string">"What was our revenue growth in Q4 vs Q3?"</span>)

<span class="kw">print</span>(result.answer)
<span class="kw">print</span>(f<span class="string">"Rewritten queries: {result.rewritten_queries}"</span>)
<span class="kw">print</span>(f<span class="string">"Sources used: {result.sources_used}"</span>)
<span class="kw">print</span>(f<span class="string">"Confidence: {result.confidence}"</span>)</code></pre>
</div>

<!-- ══════ DEEP AGENT ══════ -->

<div id="page-deep-agent" class="page">
<h1>Deep Agent Architecture</h1>
<p>Full deep agent with Virtual File System, Planning, and context quarantine for long-horizon tasks.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.deep_agent <span class="kw">import</span> DeepAgent
<span class="kw">from</span> duxx_ai.core.agent <span class="kw">import</span> Agent, AgentConfig

base = Agent(config=AgentConfig(name=<span class="string">"researcher"</span>, system_prompt=<span class="string">"..."</span>))
deep = DeepAgent(base, workspace=<span class="string">"research_project"</span>)

<span class="cm"># Agent automatically gets VFS + Planning tools</span>
result = <span class="kw">await</span> deep.run(<span class="string">"Research AI agent architectures and write a report"</span>)

<span class="cm"># Agent writes artifacts to workspace</span>
<span class="kw">print</span>(deep.workspace.tree())
<span class="cm"># analysis/report.md (2400 bytes)</span>
<span class="cm"># code/evaluation.py (800 bytes)</span>

<span class="cm"># Check plan progress</span>
<span class="kw">print</span>(deep.plan.get_status())</code></pre>
</div>

<div id="page-vfs" class="page">
<h1>Virtual File System</h1>
<p>Persistent artifact storage beyond context window limits. Agents offload large outputs to files.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.deep_agent <span class="kw">import</span> VirtualFileSystem

vfs = VirtualFileSystem(persist_path=<span class="string">"workspace"</span>)

<span class="cm"># Write artifacts</span>
vfs.write(<span class="string">"analysis/q4_report.md"</span>, <span class="string">"# Q4 Analysis\n..."</span>)
vfs.write(<span class="string">"code/model.py"</span>, <span class="string">"import torch\n..."</span>)

<span class="cm"># Read when needed</span>
content = vfs.read(<span class="string">"analysis/q4_report.md"</span>)

<span class="cm"># Search across files</span>
results = vfs.search(<span class="string">"revenue growth"</span>)

<span class="cm"># Generate agent tools</span>
tools = vfs.as_tools()  <span class="cm"># [read_file, write_file, list_files, search_files]</span></code></pre>
</div>

<div id="page-planning" class="page">
<h1>Planning Tool</h1>
<p>Externalized strategic thinking via todo_write-style interface.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.deep_agent <span class="kw">import</span> PlanningTool

planner = PlanningTool()
planner.create_plan(<span class="string">"Market Analysis"</span>, [<span class="string">"Gather data"</span>, <span class="string">"Analyze trends"</span>, <span class="string">"Write report"</span>])
planner.update_step(<span class="number">1</span>, <span class="string">"completed"</span>, <span class="string">"Downloaded Q4 data from 3 sources"</span>)
planner.update_step(<span class="number">2</span>, <span class="string">"in_progress"</span>)

<span class="kw">print</span>(planner.get_status())
<span class="cm"># ● [1] Gather data (completed) → Downloaded Q4 data from 3 sources</span>
<span class="cm"># ◑ [2] Analyze trends (in_progress)</span>
<span class="cm"># ○ [3] Write report (pending)</span>

<span class="cm"># Generate agent tools</span>
tools = planner.as_tools()  <span class="cm"># [plan_create, plan_update, plan_status, plan_next, plan_add_step]</span></code></pre>
</div>

<div id="page-graph-analytics" class="page">
<h1>Graph Analytics — 35+ NetworkX Algorithms</h1>
<p>Deep analysis of agent workflow graphs powered by NetworkX. Optimize workflows, detect bottlenecks, rank agents, predict connections, and export to any format.</p>
<p><code>pip install networkx</code> (optional — basic analysis works without it)</p>

<h2 id="ga-setup">Setup &amp; Basic Usage</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.analytics <span class="kw">import</span> WorkflowAnalyzer

<span class="cm"># From raw nodes and edges</span>
analyzer = WorkflowAnalyzer(
    nodes=[<span class="string">"fetch"</span>, <span class="string">"process_a"</span>, <span class="string">"process_b"</span>, <span class="string">"merge"</span>, <span class="string">"output"</span>],
    edges=[(<span class="string">"fetch"</span>, <span class="string">"process_a"</span>), (<span class="string">"fetch"</span>, <span class="string">"process_b"</span>),
           (<span class="string">"process_a"</span>, <span class="string">"merge"</span>), (<span class="string">"process_b"</span>, <span class="string">"merge"</span>),
           (<span class="string">"merge"</span>, <span class="string">"output"</span>)],
)

<span class="cm"># Or from a compiled FlowGraph</span>
analyzer = WorkflowAnalyzer.from_flow_graph(compiled_flow)

<span class="cm"># Full report</span>
report = analyzer.full_report()
<span class="kw">print</span>(report.summary())</code></pre>

<h2 id="ga-critical">Critical Path &amp; Bottlenecks</h2>
<pre class="code-block"><code>report = analyzer.full_report()

<span class="cm"># Critical path — longest execution chain</span>
<span class="kw">print</span>(report.critical_path)           <span class="cm"># ['fetch', 'process_a', 'merge', 'output']</span>
<span class="kw">print</span>(report.critical_path_length)     <span class="cm"># 4</span>

<span class="cm"># Bottlenecks — nodes everything flows through</span>
<span class="kw">print</span>(report.bottlenecks)              <span class="cm"># ['merge']</span>

<span class="cm"># Single points of failure — removal disconnects graph</span>
<span class="kw">print</span>(report.single_points_of_failure) <span class="cm"># ['merge']</span>

<span class="cm"># Articulation points &amp; bridges</span>
<span class="kw">print</span>(analyzer.articulation_points())  <span class="cm"># ['merge'] — critical nodes</span>
<span class="kw">print</span>(analyzer.bridges())              <span class="cm"># [('merge','output')] — critical edges</span></code></pre>

<h2 id="ga-parallel">Parallel Scheduling</h2>
<pre class="code-block"><code><span class="cm"># Find groups that can run in parallel</span>
<span class="kw">print</span>(report.parallel_opportunities)   <span class="cm"># [['process_a', 'process_b']]</span>

<span class="cm"># Topological generations — execute level by level</span>
gens = analyzer.topological_generations()
<span class="kw">for</span> i, level <span class="kw">in</span> <span class="kw">enumerate</span>(gens):
    <span class="kw">print</span>(f<span class="string">"Level {i}: {level} (run in parallel)"</span>)
<span class="cm"># Level 0: ['fetch'] (run in parallel)</span>
<span class="cm"># Level 1: ['process_a', 'process_b'] (run in parallel)</span>
<span class="cm"># Level 2: ['merge'] (run in parallel)</span>
<span class="cm"># Level 3: ['output'] (run in parallel)</span>

<span class="cm"># Graph width — maximum parallelism possible</span>
<span class="kw">print</span>(report.width)                    <span class="cm"># 2</span></code></pre>

<h2 id="ga-ranking">Node Importance Ranking</h2>
<pre class="code-block"><code><span class="cm"># PageRank — Google's algorithm applied to agents</span>
<span class="kw">print</span>(report.pagerank)
<span class="cm"># {'merge': 0.28, 'fetch': 0.22, 'output': 0.20, ...}</span>

<span class="cm"># Top nodes by importance</span>
<span class="kw">print</span>(report.most_important_nodes[:3])
<span class="cm"># [('merge', 0.28), ('fetch', 0.22), ('output', 0.20)]</span>

<span class="cm"># HITS — Hub/Authority scores</span>
hubs, authorities = analyzer.hits()
<span class="cm"># Hubs = nodes routing to many others (orchestrators)</span>
<span class="cm"># Authorities = nodes many route to (workers)</span>

<span class="cm"># Detailed per-node metrics</span>
<span class="kw">for</span> node_id, m <span class="kw">in</span> report.node_metrics.items():
    <span class="kw">print</span>(f<span class="string">"{node_id}: degree={m.degree_centrality:.2f}, betweenness={m.betweenness_centrality:.2f}"</span>)</code></pre>

<h2 id="ga-dag">DAG Analysis &amp; Dependencies</h2>
<pre class="code-block"><code><span class="cm"># Ancestors — all upstream nodes</span>
<span class="kw">print</span>(analyzer.ancestors(<span class="string">"merge"</span>))     <span class="cm"># {'fetch', 'process_a', 'process_b'}</span>

<span class="cm"># Descendants — all downstream nodes</span>
<span class="kw">print</span>(analyzer.descendants(<span class="string">"fetch"</span>))  <span class="cm"># {'process_a', 'process_b', 'merge', 'output'}</span>

<span class="cm"># Transitive reduction — remove redundant edges</span>
<span class="kw">print</span>(analyzer.transitive_reduction())
<span class="cm"># Only essential edges, redundant shortcuts removed</span>

<span class="cm"># Transitive closure — all reachable pairs</span>
<span class="kw">print</span>(analyzer.transitive_closure())

<span class="cm"># Cycle detection</span>
<span class="kw">print</span>(analyzer.find_cycles())          <span class="cm"># [] if DAG, or list of cycles</span>
<span class="kw">print</span>(report.is_dag)                   <span class="cm"># True</span></code></pre>

<h2 id="ga-flow">Flow &amp; Capacity</h2>
<pre class="code-block"><code><span class="cm"># Add edge weights for capacity</span>
analyzer = WorkflowAnalyzer(
    nodes=[<span class="string">"source"</span>, <span class="string">"a"</span>, <span class="string">"b"</span>, <span class="string">"sink"</span>],
    edges=[(<span class="string">"source"</span>,<span class="string">"a"</span>), (<span class="string">"source"</span>,<span class="string">"b"</span>), (<span class="string">"a"</span>,<span class="string">"sink"</span>), (<span class="string">"b"</span>,<span class="string">"sink"</span>)],
    edge_weights={(<span class="string">"source"</span>,<span class="string">"a"</span>): 10, (<span class="string">"source"</span>,<span class="string">"b"</span>): 5, (<span class="string">"a"</span>,<span class="string">"sink"</span>): 8, (<span class="string">"b"</span>,<span class="string">"sink"</span>): 7},
)

<span class="cm"># Maximum flow — how much throughput possible?</span>
flow_value, flow_dict = analyzer.max_flow(<span class="string">"source"</span>, <span class="string">"sink"</span>)
<span class="kw">print</span>(f<span class="string">"Max throughput: {flow_value}"</span>)  <span class="cm"># 15</span>

<span class="cm"># Minimum cut — weakest link</span>
cut_value, (set_a, set_b) = analyzer.min_cut(<span class="string">"source"</span>, <span class="string">"sink"</span>)
<span class="kw">print</span>(f<span class="string">"Min cut: {cut_value}, partition: {set_a} | {set_b}"</span>)</code></pre>

<h2 id="ga-optimization">Optimization Algorithms</h2>
<pre class="code-block"><code><span class="cm"># Graph coloring — assign resources without conflicts</span>
colors = analyzer.graph_coloring()
<span class="kw">print</span>(colors)  <span class="cm"># {'a': 0, 'b': 1, 'c': 1, 'd': 0} — same color = can share resources</span>

<span class="cm"># Dominating set — minimum nodes to monitor all others</span>
dom = analyzer.dominating_set()
<span class="kw">print</span>(dom)  <span class="cm"># {'fetch', 'merge'} — monitoring these covers everything</span>

<span class="cm"># Minimum spanning tree — most efficient connection</span>
mst = analyzer.minimum_spanning_tree()
<span class="kw">print</span>(mst)  <span class="cm"># Essential edges only</span></code></pre>

<h2 id="ga-community">Community Detection &amp; Structure</h2>
<pre class="code-block"><code><span class="cm"># Communities — group related agents</span>
<span class="kw">print</span>(report.communities)
<span class="cm"># [['fetch', 'process_a'], ['process_b', 'merge', 'output']]</span>

<span class="cm"># Strongly connected components</span>
<span class="kw">print</span>(analyzer.strongly_connected_components())

<span class="cm"># Structural holes — brokerage opportunities</span>
<span class="kw">print</span>(analyzer.structural_holes())
<span class="cm"># Lower constraint = more brokerage power between groups</span>

<span class="cm"># Clustering coefficient — how tightly connected neighborhoods are</span>
<span class="kw">print</span>(analyzer.clustering_coefficient())</code></pre>

<h2 id="ga-prediction">Link Prediction &amp; Classification</h2>
<pre class="code-block"><code><span class="cm"># Predict likely new connections between agents</span>
predictions = analyzer.link_prediction(method=<span class="string">"jaccard"</span>)
<span class="kw">for</span> u, v, score <span class="kw">in</span> predictions[:5]:
    <span class="kw">print</span>(f<span class="string">"Likely connection: {u} -> {v} (score: {score:.3f})"</span>)

<span class="cm"># Methods: "jaccard", "adamic_adar", "preferential"</span>

<span class="cm"># Node classification — propagate roles through workflow</span>
labeled = {<span class="string">"fetch"</span>: <span class="string">"data"</span>, <span class="string">"output"</span>: <span class="string">"action"</span>}
all_labels = analyzer.node_classification(labeled)
<span class="kw">print</span>(all_labels)
<span class="cm"># {'fetch': 'data', 'process_a': 'data', 'process_b': 'data',</span>
<span class="cm">#  'merge': 'data', 'output': 'action'}</span></code></pre>

<h2 id="ga-routing">Routing Algorithms</h2>
<pre class="code-block"><code><span class="cm"># Shortest path between nodes</span>
path = analyzer.shortest_path(<span class="string">"fetch"</span>, <span class="string">"output"</span>)
<span class="kw">print</span>(path)  <span class="cm"># ['fetch', 'process_a', 'merge', 'output']</span>

<span class="cm"># All possible paths</span>
all_p = analyzer.all_paths(<span class="string">"fetch"</span>, <span class="string">"output"</span>)
<span class="kw">print</span>(all_p)
<span class="cm"># [['fetch','process_a','merge','output'], ['fetch','process_b','merge','output']]</span>

<span class="cm"># Optimal (minimum weight) route</span>
optimal = analyzer.optimal_route(<span class="string">"fetch"</span>, <span class="string">"output"</span>)</code></pre>

<h2 id="ga-metrics">Graph Metrics</h2>
<pre class="code-block"><code><span class="cm"># Quality scores</span>
<span class="kw">print</span>(f<span class="string">"Connectivity: {report.connectivity_score:.2f}"</span>)  <span class="cm"># 0-1</span>
<span class="kw">print</span>(f<span class="string">"Balance: {report.balance_score:.2f}"</span>)          <span class="cm"># 0-1 (workload distribution)</span>
<span class="kw">print</span>(f<span class="string">"Complexity: {report.complexity_score:.2f}"</span>)      <span class="cm"># 0-1 (McCabe-like)</span>

<span class="cm"># Structural metrics</span>
<span class="kw">print</span>(f<span class="string">"Density: {analyzer.density():.3f}"</span>)
<span class="kw">print</span>(f<span class="string">"Diameter: {analyzer.diameter()}"</span>)
<span class="kw">print</span>(f<span class="string">"Radius: {analyzer.radius()}"</span>)
<span class="kw">print</span>(f<span class="string">"Center nodes: {analyzer.center_nodes()}"</span>)
<span class="kw">print</span>(f<span class="string">"Periphery: {analyzer.periphery_nodes()}"</span>)
<span class="kw">print</span>(f<span class="string">"Avg clustering: {analyzer.average_clustering():.3f}"</span>)
<span class="kw">print</span>(f<span class="string">"Is bipartite: {analyzer.is_bipartite()}"</span>)

<span class="cm"># Efficiency</span>
eff = analyzer.efficiency()
<span class="kw">print</span>(f<span class="string">"Local efficiency: {eff['local']:.3f}"</span>)
<span class="kw">print</span>(f<span class="string">"Global efficiency: {eff['global']:.3f}"</span>)</code></pre>

<h2 id="ga-export">Export Formats</h2>
<pre class="code-block"><code><span class="cm"># ASCII art</span>
<span class="kw">print</span>(analyzer.full_report().summary())

<span class="cm"># Mermaid diagram (paste into GitHub markdown)</span>
<span class="kw">print</span>(analyzer.to_mermaid())
<span class="cm"># graph LR</span>
<span class="cm">#     fetch --> process_a</span>
<span class="cm">#     fetch --> process_b</span>
<span class="cm">#     ...</span>

<span class="cm"># DOT format (GraphViz)</span>
<span class="kw">print</span>(analyzer.to_dot())

<span class="cm"># Standard graph formats</span>
graphml = analyzer.to_graphml()  <span class="cm"># GraphML XML</span>
gml = analyzer.to_gml()          <span class="cm"># Graph Modelling Language</span>
gexf = analyzer.to_gexf()        <span class="cm"># Gephi format</span>

<span class="cm"># JSON (for web visualization)</span>
data = analyzer.to_json()

<span class="cm"># Adjacency matrix</span>
matrix = analyzer.to_adjacency_matrix()

<span class="cm"># Get raw NetworkX graph for custom analysis</span>
G = analyzer.to_networkx()  <span class="cm"># nx.DiGraph object</span></code></pre>
</div>

<div id="page-a2a" class="page">
<h1>A2A Protocol</h1>
<p>Agent-to-Agent communication protocol for inter-agent messaging, discovery, and delegation.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.deep_agent <span class="kw">import</span> A2AProtocol

protocol = A2AProtocol()
protocol.register(<span class="string">"researcher"</span>, researcher_agent, capabilities=[<span class="string">"research"</span>, <span class="string">"analysis"</span>])
protocol.register(<span class="string">"writer"</span>, writer_agent, capabilities=[<span class="string">"writing"</span>, <span class="string">"editing"</span>])

<span class="cm"># Direct messaging</span>
response = <span class="kw">await</span> protocol.send(<span class="string">"researcher"</span>, <span class="string">"writer"</span>, <span class="string">"Please edit this draft"</span>)

<span class="cm"># Capability-based discovery</span>
agents = protocol.discover(capability=<span class="string">"research"</span>)

<span class="cm"># Broadcast to all agents</span>
responses = <span class="kw">await</span> protocol.broadcast(<span class="string">"manager"</span>, <span class="string">"New data available"</span>)

<span class="cm"># Delegate by capability</span>
result = <span class="kw">await</span> protocol.delegate(<span class="string">"manager"</span>, <span class="string">"Write Q4 summary"</span>, capability=<span class="string">"writing"</span>)</code></pre>
</div>

<!-- ══════ LANGGRAPH COMPATIBLE ══════ -->

<div id="page-lg-compile" class="page">
<h1>compile() &amp; Command</h1>
<p>LangGraph-compatible graph compilation, validation, and resume control.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.graph <span class="kw">import</span> Graph, Command, InMemorySaver

graph = Graph(<span class="string">"my-workflow"</span>)
graph.add_node(<span class="string">"step1"</span>, handler1)
graph.add_node(<span class="string">"step2"</span>, handler2)
graph.add_edge(<span class="string">"step1"</span>, <span class="string">"step2"</span>)

<span class="cm"># Compile with checkpointer</span>
graph.compile(checkpointer=InMemorySaver())

<span class="cm"># Conditional routing with function</span>
graph.add_conditional_edge(<span class="string">"analyze"</span>, route_fn, {
    <span class="string">"approve"</span>: <span class="string">"publish"</span>,
    <span class="string">"reject"</span>: <span class="string">"revise"</span>,
})

<span class="cm"># External state modification</span>
graph.update_state(
    config={<span class="string">"configurable"</span>: {<span class="string">"thread_id"</span>: <span class="string">"t1"</span>}},
    values={<span class="string">"approved"</span>: <span class="number">True</span>},
    as_node=<span class="string">"human_review"</span>,
)

<span class="cm"># Command for resume control</span>
cmd = Command(resume=<span class="string">"approved"</span>, goto=<span class="string">"publish"</span>, update={<span class="string">"status"</span>: <span class="string">"ready"</span>})</code></pre>
</div>

<div id="page-lg-interrupt" class="page">
<h1>Interrupts &amp; HITL</h1>
<p>Pause graph execution for human input with state preservation.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.graph <span class="kw">import</span> Graph, GraphInterrupt, interrupt

<span class="kw">async def</span> review_node(state):
    <span class="cm"># Pause for human approval</span>
    interrupt(<span class="string">"Do you approve this action?"</span>)
    state.set(<span class="string">"approved"</span>, <span class="number">True</span>)
    <span class="kw">return</span> state

graph = Graph(<span class="string">"approval-flow"</span>)
graph.add_node(<span class="string">"review"</span>, review_node, node_type=<span class="string">"HUMAN"</span>)

<span class="cm"># Run — will pause at review node</span>
<span class="kw">try</span>:
    result = <span class="kw">await</span> graph.run()
<span class="kw">except</span> GraphInterrupt <span class="kw">as</span> e:
    <span class="kw">print</span>(f<span class="string">"Paused: {e.prompt}"</span>)
    <span class="cm"># Resume with human input</span>
    result = <span class="kw">await</span> graph.resume(human_input={<span class="string">"decision"</span>: <span class="string">"approved"</span>})</code></pre>
</div>

<div id="page-lg-checkpoints" class="page">
<h1>Checkpointers</h1>
<p>Pluggable checkpoint backends for state persistence.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.graph <span class="kw">import</span> (
    InMemorySaver, SqliteCheckpointer, BaseCheckpointSaver
)

<span class="cm"># In-memory (development)</span>
graph.compile(checkpointer=InMemorySaver())

<span class="cm"># SQLite (persistent)</span>
graph.compile(checkpointer=SqliteCheckpointer(<span class="string">"checkpoints.db"</span>))

<span class="cm"># Custom backend (extend BaseCheckpointSaver)</span>
<span class="kw">class</span> RedisCheckpointer(BaseCheckpointSaver):
    <span class="kw">async def</span> put(self, config, state): ...
    <span class="kw">async def</span> get(self, config): ...
    <span class="kw">async def</span> list(self, config): ...

<span class="cm"># State access</span>
state = graph.get_state(config={<span class="string">"configurable"</span>: {<span class="string">"thread_id"</span>: <span class="string">"t1"</span>}})
history = graph.get_state_history()  <span class="cm"># Reverse chronological</span></code></pre>
</div>

<div id="page-lg-task" class="page">
<h1>@task &amp; Durable Execution</h1>
<p>Decorator for wrapping non-deterministic or side-effect operations in graph nodes.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.graph <span class="kw">import</span> task

<span class="cm"># Simple task</span>
@task
<span class="kw">async def</span> call_api(url: str) -> dict:
    response = <span class="kw">await</span> httpx.get(url)
    <span class="kw">return</span> response.json()

<span class="cm"># Task with retries</span>
@task(retries=<span class="number">3</span>)
<span class="kw">async def</span> send_email(to: str, body: str) -> bool:
    <span class="cm"># Will retry up to 3 times with backoff on failure</span>
    ...

<span class="cm"># Use in graph node</span>
<span class="kw">async def</span> my_node(state):
    data = <span class="kw">await</span> call_api(<span class="string">"https://api.example.com/data"</span>)
    state.set(<span class="string">"api_data"</span>, data)
    <span class="kw">return</span> state</code></pre>
</div>

<div id="page-lg-streaming" class="page">
<h1>Streaming Modes</h1>
<p>Stream graph execution with different modes for real-time UI updates.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration <span class="kw">import</span> Graph

graph = Graph(<span class="string">"pipeline"</span>)

<span class="cm"># Stream with "updates" mode (state diffs per node)</span>
<span class="kw">async for</span> event <span class="kw">in</span> graph.stream(initial_state, mode=<span class="string">"updates"</span>):
    <span class="kw">print</span>(f<span class="string">"Node: {event['node']}, Changes: {event['changes']}"</span>)

<span class="cm"># Stream with "values" mode (full state after each node)</span>
<span class="kw">async for</span> event <span class="kw">in</span> graph.stream(initial_state, mode=<span class="string">"values"</span>):
    <span class="kw">print</span>(f<span class="string">"State: {event['state']}"</span>)

<span class="cm"># Stream with "debug" mode (includes timing and iteration info)</span>
<span class="kw">async for</span> event <span class="kw">in</span> graph.stream(initial_state, mode=<span class="string">"debug"</span>):
    <span class="kw">print</span>(f<span class="string">"Node: {event['node']}, Duration: {event['duration_ms']}ms"</span>)</code></pre>
</div>

<!-- ══════ IMPORTERS ══════ -->

<div id="page-n8n-import" class="page">
<h1>n8n Workflow Import</h1>
<p>Import n8n workflow JSON files and convert to Duxx AI graph code.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.importers.n8n <span class="kw">import</span> N8nImporter

importer = N8nImporter()

<span class="cm"># Load n8n JSON</span>
<span class="kw">with</span> open(<span class="string">"workflow.json"</span>) <span class="kw">as</span> f:
    n8n_json = json.load(f)

<span class="cm"># Convert to Duxx AI</span>
result = importer.convert(n8n_json)

<span class="kw">print</span>(result[<span class="string">"pattern"</span>])       <span class="cm"># "graph" or "crew"</span>
<span class="kw">print</span>(result[<span class="string">"python_code"</span>])   <span class="cm"># Ready-to-run Duxx AI code</span>
<span class="kw">print</span>(result[<span class="string">"node_count"</span>])    <span class="cm"># Number of nodes converted</span>
<span class="kw">print</span>(result[<span class="string">"warnings"</span>])      <span class="cm"># Any conversion warnings</span></code></pre>
</div>

<!-- ── MCP Integration ── -->
<div id="page-mcp" class="page">
<h1>MCP Integration</h1>
<p>Connect to any MCP server or expose Duxx AI tools as an MCP server.</p>

<h2 id="mcp-client">MCPClient — Connect to MCP Servers</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.tools.mcp <span class="kw">import</span> MCPClient

<span class="cm"># stdio transport (local server)</span>
<span class="kw">async with</span> MCPClient(<span class="string">"stdio"</span>, command=<span class="string">"python"</span>, args=[<span class="string">"math_server.py"</span>]) <span class="kw">as</span> client:
    tools = <span class="kw">await</span> client.load_tools()    <span class="cm"># Native Duxx AI Tool objects</span>
    result = <span class="kw">await</span> client.call_tool(<span class="string">"add"</span>, {<span class="string">"a"</span>: 1, <span class="string">"b"</span>: 2})

<span class="cm"># HTTP transport (remote server)</span>
<span class="kw">async with</span> MCPClient(<span class="string">"http"</span>, url=<span class="string">"http://localhost:8000/mcp"</span>,
                     headers={<span class="string">"Authorization"</span>: <span class="string">"Bearer token"</span>}) <span class="kw">as</span> client:
    tools = <span class="kw">await</span> client.load_tools()
    resources = <span class="kw">await</span> client.list_resources()
    prompts = <span class="kw">await</span> client.list_prompts()</code></pre>

<h2 id="mcp-toolkit">MCPToolkit — Multi-Server Management</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.tools.mcp <span class="kw">import</span> MCPToolkit

toolkit = MCPToolkit({
    <span class="string">"math"</span>: {<span class="string">"transport"</span>: <span class="string">"stdio"</span>, <span class="string">"command"</span>: <span class="string">"python"</span>, <span class="string">"args"</span>: [<span class="string">"math.py"</span>]},
    <span class="string">"weather"</span>: {<span class="string">"transport"</span>: <span class="string">"http"</span>, <span class="string">"url"</span>: <span class="string">"http://localhost:8000/mcp"</span>},
    <span class="string">"db"</span>: {<span class="string">"transport"</span>: <span class="string">"http"</span>, <span class="string">"url"</span>: <span class="string">"https://api.example.com/mcp"</span>,
           <span class="string">"headers"</span>: {<span class="string">"Authorization"</span>: <span class="string">"Bearer key"</span>}},
})

<span class="kw">async with</span> toolkit:
    all_tools = <span class="kw">await</span> toolkit.get_tools()       <span class="cm"># From all servers</span>
    math_tools = <span class="kw">await</span> toolkit.get_tools(<span class="string">"math"</span>)  <span class="cm"># From specific server</span>

    <span class="cm"># Use with Agent</span>
    agent = Agent(tools=all_tools, config=AgentConfig(name=<span class="string">"mcp-agent"</span>))
    result = <span class="kw">await</span> agent.run(<span class="string">"What is 2+2?"</span>)</code></pre>

<h2 id="mcp-server">MCPServer — Expose Duxx AI Tools</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.tools.mcp <span class="kw">import</span> MCPServer
<span class="kw">from</span> duxx_ai.tools.builtin <span class="kw">import</span> get_builtin_tools

<span class="cm"># Expose tools as MCP server (works with Claude Desktop, Cursor, etc.)</span>
tools = get_builtin_tools([<span class="string">"calculator"</span>, <span class="string">"web_request"</span>])
server = MCPServer(<span class="string">"duxx-agent"</span>, tools=tools)
server.run(transport=<span class="string">"stdio"</span>)  <span class="cm"># or "sse", port=8000</span></code></pre>
</div>

<!-- ── Middleware ── -->
<div id="page-middleware" class="page">
<h1>Middleware</h1>
<p>Wrap LLM calls with cross-cutting concerns: caching, moderation, logging, rate limiting.</p>

<h2 id="mw-cache">Prompt Cache</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.middleware <span class="kw">import</span> PromptCacheMiddleware

cache = PromptCacheMiddleware(ttl_seconds=300, max_entries=1000)
<span class="cm"># Caches identical prompts for 5 minutes</span>
<span class="kw">print</span>(cache.stats)  <span class="cm"># {"hits": 42, "misses": 58, "size": 58}</span></code></pre>

<h2 id="mw-moderation">Content Moderation</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.middleware <span class="kw">import</span> ContentModerationMiddleware

mod = ContentModerationMiddleware(
    block_pii=<span class="kw">True</span>,         <span class="cm"># Detect SSN, CC, email, phone</span>
    block_injection=<span class="kw">True</span>,   <span class="cm"># Block prompt injection</span>
    block_profanity=<span class="kw">False</span>,  <span class="cm"># Optional profanity filter</span>
    action=<span class="string">"redact"</span>,         <span class="cm"># "block", "redact", or "warn"</span>
)</code></pre>

<h2 id="mw-chain">Middleware Chain</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.middleware <span class="kw">import</span> (
    MiddlewareChain, PromptCacheMiddleware,
    ContentModerationMiddleware, LoggingMiddleware, RateLimitMiddleware,
)

chain = MiddlewareChain([
    RateLimitMiddleware(max_calls_per_minute=60),
    PromptCacheMiddleware(ttl_seconds=300),
    ContentModerationMiddleware(block_pii=<span class="kw">True</span>),
    LoggingMiddleware(log_prompts=<span class="kw">True</span>),
])

<span class="cm"># Apply before/after every LLM call</span>
metadata = <span class="kw">await</span> chain.before_call(conversation, system_prompt, {})
<span class="kw">if not</span> metadata.get(<span class="string">"_blocked"</span>):
    response = <span class="kw">await</span> provider.complete(conversation)
    response = <span class="kw">await</span> chain.after_call(response.content, metadata)</code></pre>
</div>

<!-- ── Integration Catalog ── -->
<div id="page-integrations" class="page">
<h1>Integration Catalog — 490 Integrations</h1>
<p>Duxx AI supports <strong>490 integrations</strong> across 9 categories, exceeding LangChain in every category. All providers use <strong>lazy imports</strong> — install only the packages you need.</p>

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0;">
<div style="background:var(--blue-bg);padding:16px;border-radius:8px;text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--blue);">72</div><div style="font-size:12px;color:var(--gray-600);">Chat Models</div></div>
<div style="background:var(--purple-bg);padding:16px;border-radius:8px;text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--purple);">86</div><div style="font-size:12px;color:var(--gray-600);">Embeddings</div></div>
<div style="background:var(--green-bg);padding:16px;border-radius:8px;text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--green);">153</div><div style="font-size:12px;color:var(--gray-600);">Vector Stores</div></div>
<div style="background:var(--orange-bg);padding:16px;border-radius:8px;text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--orange);">63</div><div style="font-size:12px;color:var(--gray-600);">Doc Loaders</div></div>
<div style="background:var(--pink-bg);padding:16px;border-radius:8px;text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--pink);">94</div><div style="font-size:12px;color:var(--gray-600);">Retrievers</div></div>
<div style="background:var(--navy-50);padding:16px;border-radius:8px;text-align:center;"><div style="font-size:28px;font-weight:800;color:var(--navy);">22</div><div style="font-size:12px;color:var(--gray-600);">Parsers + Middleware + Checkpointers</div></div>
</div>

<h2>Universal Pattern</h2>
<p>Every integration follows the same pattern — import, configure, use:</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> LLMConfig, create_provider

<span class="cm"># 1. Pick any of 72 providers</span>
provider = create_provider(LLMConfig(
    provider=<span class="string">"openai"</span>,           <span class="cm"># or "anthropic", "groq", "mistral", ...</span>
    model=<span class="string">"gpt-4o"</span>,
    api_key=<span class="string">"sk-..."</span>,            <span class="cm"># or set OPENAI_API_KEY env var</span>
))

<span class="cm"># 2. Use with Agent</span>
<span class="kw">from</span> duxx_ai <span class="kw">import</span> Agent, AgentConfig
agent = Agent(config=AgentConfig(name=<span class="string">"my-agent"</span>, llm=LLMConfig(provider=<span class="string">"groq"</span>, model=<span class="string">"llama-3.3-70b-versatile"</span>)))
result = <span class="kw">await</span> agent.run(<span class="string">"Hello!"</span>)</code></pre>
<p>Click the sub-pages in the sidebar to see every provider with detailed snippets.</p>

<h2 id="int-chat">Chat Models (72 providers)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> LLMConfig, create_provider

<span class="cm"># OpenAI</span>
p = create_provider(LLMConfig(provider=<span class="string">"openai"</span>, model=<span class="string">"gpt-4o"</span>))
<span class="cm"># Anthropic</span>
p = create_provider(LLMConfig(provider=<span class="string">"anthropic"</span>, model=<span class="string">"claude-sonnet-4-20250514"</span>))
<span class="cm"># Google Gemini</span>
p = create_provider(LLMConfig(provider=<span class="string">"google"</span>, model=<span class="string">"gemini-2.0-flash"</span>))
<span class="cm"># Groq (ultra-fast)</span>
p = create_provider(LLMConfig(provider=<span class="string">"groq"</span>, model=<span class="string">"llama-3.3-70b-versatile"</span>))
<span class="cm"># DeepSeek</span>
p = create_provider(LLMConfig(provider=<span class="string">"deepseek"</span>, model=<span class="string">"deepseek-chat"</span>))
<span class="cm"># AWS Bedrock</span>
p = create_provider(LLMConfig(provider=<span class="string">"bedrock"</span>, model=<span class="string">"anthropic.claude-3-sonnet"</span>))
<span class="cm"># Ollama (local)</span>
p = create_provider(LLMConfig(provider=<span class="string">"ollama"</span>, model=<span class="string">"llama3"</span>))

<span class="cm"># All 72: openai, anthropic, local, google, gemini, bedrock, groq, mistral,</span>
<span class="cm"># deepseek, together, fireworks, cohere, perplexity, xai, cerebras,</span>
<span class="cm"># sambanova, ai21, nvidia, anyscale, openrouter, lepton, replicate,</span>
<span class="cm"># ollama, lmstudio, vllm, huggingface, cloudflare, moonshot, zhipu,</span>
<span class="cm"># qwen, yi, nebius, deepinfra, azure_openai, baichuan, minimax,</span>
<span class="cm"># stepfun, spark, volcengine, dashscope, qianfan, hunyuan, doubao,</span>
<span class="cm"># glm, konko, friendli, novita, featherless, pipeshift, runpod,</span>
<span class="cm"># modal, baseten, llama_api, maritalk, writer, aimlapi, edenai,</span>
<span class="cm"># predictionguard, gradient, oci_genai, watsonx, snowflake_cortex,</span>
<span class="cm"># databricks, sagemaker, upstage, reka, cohere_command, dappier,</span>
<span class="cm"># greennode, aleph_alpha, netmind, abso</span></code></pre>

<h2 id="int-embed">Embeddings (86 providers)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> (
    OpenAIEmbedder, HuggingFaceEmbedder, CohereEmbedder,
    GoogleEmbedder, NVIDIAEmbedder, JinaEmbedder, VoyageEmbedder,
    FastEmbedEmbedder, LocalEmbedder, OllamaEmbedder,
)

<span class="cm"># Cloud API</span>
emb = OpenAIEmbedder(model=<span class="string">"text-embedding-3-small"</span>)
emb = CohereEmbedder(model=<span class="string">"embed-english-v3.0"</span>)
emb = JinaEmbedder(model=<span class="string">"jina-embeddings-v3"</span>)

<span class="cm"># Local (free, no API key)</span>
emb = HuggingFaceEmbedder(<span class="string">"all-MiniLM-L6-v2"</span>)
emb = FastEmbedEmbedder(<span class="string">"BAAI/bge-small-en-v1.5"</span>)
emb = OllamaEmbedder()
emb = LocalEmbedder()  <span class="cm"># Hash-based (testing only)</span>

vec = emb.embed(<span class="string">"Hello world"</span>)      <span class="cm"># list[float]</span>
vecs = emb.embed_many([<span class="string">"a"</span>, <span class="string">"b"</span>])  <span class="cm"># list[list[float]]</span></code></pre>

<h2 id="int-vs">Vector Stores (153 backends)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.vectorstore <span class="kw">import</span> (
    InMemoryVectorStore, FAISSVectorStore, ChromaVectorStore,
    PineconeVectorStore, QdrantVectorStore, WeaviateVectorStore,
    MilvusVectorStore, ElasticsearchVectorStore, PGVectorStore,
    LanceDBVectorStore, RedisVectorStore, MongoDBAtlasVectorStore,
)

<span class="cm"># In-memory (no deps)</span>
store = InMemoryVectorStore(embedder)

<span class="cm"># FAISS (pip install faiss-cpu)</span>
store = FAISSVectorStore(embedder, dimension=1536)
store.save(<span class="string">"index.faiss"</span>)  <span class="cm"># Persist to disk</span>

<span class="cm"># ChromaDB (pip install chromadb)</span>
store = ChromaVectorStore(embedder, collection_name=<span class="string">"docs"</span>,
                          persist_directory=<span class="string">"./chroma_db"</span>)

<span class="cm"># Pinecone (cloud)</span>
store = PineconeVectorStore(embedder, index_name=<span class="string">"my-index"</span>)

<span class="cm"># Common API for all stores:</span>
ids = store.add(documents)
results = store.search(<span class="string">"query"</span>, top_k=5)  <span class="cm"># list[SearchResult]</span>
store.delete(ids)
count = store.count()</code></pre>

<h2 id="int-loaders">Document Loaders (63 types)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.loaders <span class="kw">import</span> (
    TextLoader, PDFLoader, CSVLoader, MarkdownLoader,
    WebLoader, GitHubLoader, NotionLoader, WikipediaLoader,
    S3Loader, ExcelLoader, SlackLoader, YouTubeLoader,
    DirectoryLoader, RecursiveURLLoader,
)

<span class="cm"># Files</span>
docs = TextLoader(<span class="string">"readme.txt"</span>).load()
docs = PDFLoader(<span class="string">"paper.pdf"</span>).load()
docs = ExcelLoader(<span class="string">"data.xlsx"</span>).load()
docs = MarkdownLoader(<span class="string">"docs.md"</span>, split_by_headers=<span class="kw">True</span>).load()

<span class="cm"># Web</span>
docs = WebLoader(<span class="string">"https://example.com"</span>).load()
docs = RecursiveURLLoader(<span class="string">"https://docs.example.com"</span>, max_depth=2).load()
docs = WikipediaLoader(<span class="string">"machine learning"</span>).load()
docs = YouTubeLoader(<span class="string">"https://youtube.com/watch?v=..."</span>).load()

<span class="cm"># Cloud</span>
docs = S3Loader(<span class="string">"my-bucket"</span>, prefix=<span class="string">"docs/"</span>).load()
docs = GitHubLoader(<span class="string">"owner/repo"</span>, extensions=[<span class="string">".md"</span>]).load()
docs = NotionLoader(page_ids=[<span class="string">"page-id"</span>]).load()
docs = SlackLoader(channel_id=<span class="string">"C123"</span>).load()

<span class="cm"># Batch directory</span>
docs = DirectoryLoader(<span class="string">"./data"</span>, glob=<span class="string">"**/*.md"</span>).load()</code></pre>

<h2 id="int-ret">Retrievers (94 types)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.retriever <span class="kw">import</span> (
    VectorRetriever, BM25Retriever, HybridRetriever,
    RerankerRetriever, MultiQueryRetriever, EnsembleRetriever,
    TavilyRetriever, WikipediaRetriever, ArxivRetriever,
    ContextualCompressionRetriever, MaxMarginalRelevanceRetriever,
)

<span class="cm"># Vector similarity</span>
ret = VectorRetriever(store, min_score=0.5)

<span class="cm"># BM25 (no deps, local)</span>
ret = BM25Retriever(documents)

<span class="cm"># Hybrid (vector + keyword with RRF fusion)</span>
ret = HybridRetriever(vector_ret, keyword_ret, vector_weight=0.6)

<span class="cm"># Reranker (Cohere or local CrossEncoder)</span>
ret = RerankerRetriever(base_ret, method=<span class="string">"cohere"</span>)

<span class="cm"># Multi-query (expand query for better recall)</span>
ret = MultiQueryRetriever(base_ret, query_count=3)

<span class="cm"># Ensemble (combine any retrievers)</span>
ret = EnsembleRetriever([vector_ret, bm25_ret, wiki_ret], weights=[0.5, 0.3, 0.2])

<span class="cm"># Web search</span>
ret = TavilyRetriever()

docs = ret.retrieve(<span class="string">"query"</span>, top_k=5)  <span class="cm"># list[Document]</span></code></pre>

<h2 id="int-check">Checkpointers (8 backends)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> (
    MemorySnapshotStore, SQLiteSnapshotStore, FileSnapshotStore,
    PostgresSnapshotStore, RedisSnapshotStore, MongoSnapshotStore,
    DynamoDBSnapshotStore, ValleySnapshotStore,
)

<span class="cm"># In-memory (default)</span>
cp = MemorySnapshotStore()
<span class="cm"># SQLite (persistent)</span>
cp = SQLiteSnapshotStore(<span class="string">"checkpoints.db"</span>)
<span class="cm"># PostgreSQL</span>
cp = PostgresSnapshotStore(<span class="string">"postgresql://user:pass@host/db"</span>)
<span class="cm"># Redis</span>
cp = RedisSnapshotStore(<span class="string">"redis://localhost:6379"</span>)
<span class="cm"># MongoDB</span>
cp = MongoSnapshotStore(<span class="string">"mongodb://localhost:27017"</span>)
<span class="cm"># DynamoDB</span>
cp = DynamoDBSnapshotStore(<span class="string">"my-table"</span>, region=<span class="string">"us-east-1"</span>)

<span class="cm"># Use with FlowGraph</span>
compiled = graph.compile(checkpointer=cp)</code></pre>

<h2 id="int-parsers">Output Parsers (10 types)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.parsers <span class="kw">import</span> (
    JSONOutputParser, PydanticOutputParser, MarkdownOutputParser,
    RegexOutputParser, ListOutputParser, RetryParser,
    XMLOutputParser, YAMLOutputParser, CSVOutputParser, EnumOutputParser,
)

<span class="cm"># JSON</span>
parser = JSONOutputParser()
result = parser.parse(<span class="string">'{"name": "Alice", "age": 30}'</span>)

<span class="cm"># Pydantic schema validation</span>
<span class="kw">class</span> User(BaseModel): name: <span class="kw">str</span>; age: <span class="kw">int</span>
parser = PydanticOutputParser(User)

<span class="cm"># XML</span>
parser = XMLOutputParser(tags=[<span class="string">"name"</span>, <span class="string">"age"</span>])
result = parser.parse(<span class="string">"&lt;name&gt;Alice&lt;/name&gt;&lt;age&gt;30&lt;/age&gt;"</span>)

<span class="cm"># Enum (constrained choices)</span>
parser = EnumOutputParser(choices=[<span class="string">"positive"</span>, <span class="string">"negative"</span>, <span class="string">"neutral"</span>])

<span class="cm"># Retry wrapper (auto-retry on parse failure)</span>
parser = RetryParser(JSONOutputParser(), max_retries=3)</code></pre>
</div>

<!-- ── FlowGraph (Updated) ── -->
<div id="page-flowgraph" class="page">
<h1>FlowGraph — Modern State Graph</h1>
<p>Typed state graph with channels, streaming, time-travel, and checkpointing.</p>

<h2 id="fg-basic">Basic FlowGraph</h2>
<pre class="code-block"><code><span class="kw">from</span> typing <span class="kw">import</span> TypedDict, Annotated
<span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> (
    FlowGraph, ENTRY, EXIT, ChatState, merge_messages,
    MemorySnapshotStore, EventMode,
)

<span class="kw">class</span> State(TypedDict):
    messages: Annotated[list, merge_messages]
    count: <span class="kw">int</span>

<span class="kw">def</span> agent(state):
    <span class="kw">return</span> {<span class="string">"count"</span>: state[<span class="string">"count"</span>] + 1}

graph = FlowGraph(State)
graph.add_node(<span class="string">"agent"</span>, agent)
graph.add_edge(ENTRY, <span class="string">"agent"</span>)
graph.add_edge(<span class="string">"agent"</span>, EXIT)

compiled = graph.compile(checkpointer=MemorySnapshotStore())
result = <span class="kw">await</span> compiled.invoke({<span class="string">"messages"</span>: [], <span class="string">"count"</span>: 0})</code></pre>

<h2 id="fg-routing">Dynamic Routing</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> Route, Dispatch

<span class="cm"># Route — jump to node with state update</span>
<span class="kw">def</span> router(state):
    <span class="kw">if</span> state[<span class="string">"urgent"</span>]:
        <span class="kw">return</span> Route(update={<span class="string">"priority"</span>: <span class="string">"high"</span>}, goto=<span class="string">"fast_track"</span>)
    <span class="kw">return</span> Route(goto=<span class="string">"normal"</span>)

<span class="cm"># Dispatch — fan-out to multiple nodes</span>
<span class="kw">def</span> splitter(state):
    <span class="kw">return</span> [Dispatch(<span class="string">"worker"</span>, {<span class="string">"item"</span>: x}) <span class="kw">for</span> x <span class="kw">in</span> state[<span class="string">"items"</span>]]</code></pre>

<h2 id="fg-hitl">Human-in-the-Loop</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> pause, FlowPause

<span class="kw">def</span> review_node(state):
    <span class="kw">if not</span> state.get(<span class="string">"approved"</span>):
        pause(<span class="string">"Please review and approve"</span>)
    <span class="kw">return</span> {<span class="string">"status"</span>: <span class="string">"approved"</span>}

<span class="cm"># Will raise FlowPause — resume with:</span>
result = <span class="kw">await</span> compiled.resume({<span class="string">"approved"</span>: <span class="kw">True</span>})</code></pre>

<h2 id="fg-stream">Streaming</h2>
<pre class="code-block"><code><span class="kw">async for</span> event <span class="kw">in</span> compiled.stream(input_data, stream_mode=EventMode.UPDATES):
    <span class="kw">print</span>(f<span class="string">"Node: {event.node}, Data: {event.data}"</span>)

<span class="cm"># Available modes: VALUES, UPDATES, MESSAGES, DEBUG, TASKS, CUSTOM</span></code></pre>

<h2 id="fg-timetravel">Time-Travel Debugging</h2>
<pre class="code-block"><code><span class="cm"># Browse checkpoint history</span>
history = <span class="kw">await</span> compiled.get_state_history(limit=20)

<span class="cm"># Replay from any checkpoint</span>
result = <span class="kw">await</span> compiled.replay_from(snapshot_id=<span class="string">"abc123"</span>)

<span class="cm"># Fork with modifications</span>
result = <span class="kw">await</span> compiled.fork_from(<span class="string">"abc123"</span>, updates={<span class="string">"count"</span>: 99})</code></pre>

<h2 id="fg-graph-view">Graph Introspection</h2>
<pre class="code-block"><code>view = compiled.get_graph()
<span class="kw">print</span>(view.to_ascii())    <span class="cm"># ASCII art diagram</span>
<span class="kw">print</span>(view.to_mermaid())   <span class="cm"># Mermaid diagram syntax</span></code></pre>

<h2 id="fg-decorators">@workflow / @step</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> workflow, step, RetryStrategy

@step(name=<span class="string">"fetch"</span>, retry_policy=RetryStrategy(max_attempts=3))
<span class="kw">async def</span> fetch_data(url: <span class="kw">str</span>) -> dict:
    <span class="kw">return</span> {<span class="string">"data"</span>: <span class="string">"..."</span>}

@workflow(checkpointer=MemorySnapshotStore())
<span class="kw">async def</span> my_pipeline(input_data: dict) -> <span class="kw">str</span>:
    data = <span class="kw">await</span> fetch_data(input_data[<span class="string">"url"</span>])
    <span class="kw">return</span> data[<span class="string">"data"</span>]

result = <span class="kw">await</span> my_pipeline({<span class="string">"url"</span>: <span class="string">"https://example.com"</span>})</code></pre>

<h2 id="fg-store">Store (Key-Value Persistence)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> Store

store = Store()
store.put(<span class="string">"user:123"</span>, <span class="string">"prefs"</span>, {<span class="string">"theme"</span>: <span class="string">"dark"</span>})
prefs = store.get(<span class="string">"user:123"</span>, <span class="string">"prefs"</span>)
results = store.search(<span class="string">"user:123"</span>, query=<span class="string">"theme"</span>)</code></pre>

<h2 id="fg-validate">Graph Validation</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> validate_graph

errors = validate_graph(graph)
<span class="kw">if</span> errors:
    <span class="kw">for</span> e <span class="kw">in</span> errors: <span class="kw">print</span>(f<span class="string">"Error: {e}"</span>)</code></pre>
</div>

<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
<!-- DETAILED INTEGRATION PAGES -->
<!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->

<!-- ── Chat Models Detail ── -->
<div id="page-int-chat-detail" class="page">
<h1>Chat Models — 72 Providers</h1>
<p>Duxx AI supports 72 LLM providers through a unified <code>create_provider()</code> API. Set the <code>provider</code> name and <code>model</code>, and Duxx AI handles the rest. API keys are read from environment variables automatically.</p>

<h2>Tier 1 — Native Implementations</h2>
<p>These providers have full custom implementations with streaming, tool calling, and provider-specific features.</p>

<h3>OpenAI</h3>
<p>The most widely used LLM API. Supports GPT-4o, GPT-4o-mini, o1, o3, and all OpenAI models.</p>
<pre class="code-block"><code><span class="cm"># pip install httpx</span>
<span class="cm"># Set: OPENAI_API_KEY=sk-...</span>

<span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> LLMConfig, create_provider

provider = create_provider(LLMConfig(
    provider=<span class="string">"openai"</span>,
    model=<span class="string">"gpt-4o"</span>,              <span class="cm"># or "gpt-4o-mini", "o1", "o3-mini"</span>
    temperature=0.7,
    max_tokens=4096,
))

<span class="cm"># Streaming</span>
<span class="kw">async for</span> chunk <span class="kw">in</span> provider.stream(conversation):
    <span class="kw">print</span>(chunk, end=<span class="string">""</span>)

<span class="cm"># With tool calling</span>
response = <span class="kw">await</span> provider.complete(conversation, tools=[calculator, search])</code></pre>

<h3>Anthropic</h3>
<p>Claude models. Supports Claude Sonnet, Haiku, and Opus with the Anthropic Messages API format.</p>
<pre class="code-block"><code><span class="cm"># Set: ANTHROPIC_API_KEY=sk-ant-...</span>

provider = create_provider(LLMConfig(
    provider=<span class="string">"anthropic"</span>,
    model=<span class="string">"claude-sonnet-4-20250514"</span>,  <span class="cm"># or "claude-haiku", "claude-opus"</span>
))

<span class="cm"># Anthropic handles system prompts separately (automatic)</span>
response = <span class="kw">await</span> provider.complete(conversation, system_prompt=<span class="string">"You are helpful."</span>)</code></pre>

<h3>Google Gemini</h3>
<p>Google's Gemini models with native function calling and streaming. Supports Gemini 2.0, 1.5 Pro/Flash.</p>
<pre class="code-block"><code><span class="cm"># Set: GOOGLE_API_KEY=AIza...</span>

provider = create_provider(LLMConfig(
    provider=<span class="string">"google"</span>,              <span class="cm"># or "gemini" (alias)</span>
    model=<span class="string">"gemini-2.0-flash"</span>,       <span class="cm"># or "gemini-1.5-pro"</span>
))</code></pre>

<h3>AWS Bedrock</h3>
<p>Access Claude, Llama, Titan, and other models via AWS Bedrock. Uses boto3 with AWS credentials.</p>
<pre class="code-block"><code><span class="cm"># pip install boto3</span>
<span class="cm"># Uses AWS credentials from ~/.aws/credentials or env vars</span>

provider = create_provider(LLMConfig(
    provider=<span class="string">"bedrock"</span>,
    model=<span class="string">"anthropic.claude-3-sonnet-20240229-v1:0"</span>,
    extra={<span class="string">"region"</span>: <span class="string">"us-east-1"</span>},
))</code></pre>

<h2>Tier 2 — OpenAI-Compatible (Major Providers)</h2>
<p>These providers use OpenAI-compatible APIs. Duxx AI automatically sets the correct base URL and auth.</p>

<h3>Groq — Ultra-Fast Inference</h3>
<pre class="code-block"><code><span class="cm"># Set: GROQ_API_KEY=gsk-...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"groq"</span>, model=<span class="string">"llama-3.3-70b-versatile"</span>))
<span class="cm"># Models: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it</span></code></pre>

<h3>Mistral AI</h3>
<pre class="code-block"><code><span class="cm"># Set: MISTRAL_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"mistral"</span>, model=<span class="string">"mistral-large-latest"</span>))
<span class="cm"># Models: mistral-large-latest, mistral-medium, mistral-small, open-mistral-nemo</span></code></pre>

<h3>DeepSeek</h3>
<pre class="code-block"><code><span class="cm"># Set: DEEPSEEK_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"deepseek"</span>, model=<span class="string">"deepseek-chat"</span>))
<span class="cm"># Models: deepseek-chat, deepseek-coder, deepseek-reasoner</span></code></pre>

<h3>Together AI — Open Source Models</h3>
<pre class="code-block"><code><span class="cm"># Set: TOGETHER_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"together"</span>, model=<span class="string">"meta-llama/Llama-3-70b-chat-hf"</span>))
<span class="cm"># 100+ open source models: Llama, Mixtral, Qwen, CodeLlama, etc.</span></code></pre>

<h3>Fireworks AI</h3>
<pre class="code-block"><code><span class="cm"># Set: FIREWORKS_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"fireworks"</span>, model=<span class="string">"accounts/fireworks/models/llama-v3p1-70b-instruct"</span>))</code></pre>

<h3>Cohere</h3>
<pre class="code-block"><code><span class="cm"># Set: COHERE_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"cohere"</span>, model=<span class="string">"command-r-plus"</span>))
<span class="cm"># Models: command-r-plus, command-r, command-light</span></code></pre>

<h3>Perplexity — Search-Augmented</h3>
<pre class="code-block"><code><span class="cm"># Set: PERPLEXITY_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"perplexity"</span>, model=<span class="string">"llama-3.1-sonar-huge-128k-online"</span>))
<span class="cm"># Built-in web search, no RAG needed</span></code></pre>

<h3>xAI (Grok)</h3>
<pre class="code-block"><code><span class="cm"># Set: XAI_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"xai"</span>, model=<span class="string">"grok-2"</span>))</code></pre>

<h3>NVIDIA NIM</h3>
<pre class="code-block"><code><span class="cm"># Set: NVIDIA_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"nvidia"</span>, model=<span class="string">"meta/llama-3.1-405b-instruct"</span>))</code></pre>

<h3>OpenRouter — 200+ Models, One API</h3>
<pre class="code-block"><code><span class="cm"># Set: OPENROUTER_API_KEY=...</span>
provider = create_provider(LLMConfig(provider=<span class="string">"openrouter"</span>, model=<span class="string">"anthropic/claude-3.5-sonnet"</span>))
<span class="cm"># Access any model from any provider through OpenRouter</span></code></pre>

<h3>Local Models — Ollama, LM Studio, vLLM</h3>
<pre class="code-block"><code><span class="cm"># Ollama (no API key, runs locally)</span>
provider = create_provider(LLMConfig(provider=<span class="string">"ollama"</span>, model=<span class="string">"llama3"</span>))

<span class="cm"># LM Studio (local GUI)</span>
provider = create_provider(LLMConfig(provider=<span class="string">"lmstudio"</span>, model=<span class="string">"local-model"</span>))

<span class="cm"># vLLM (high-throughput server)</span>
provider = create_provider(LLMConfig(provider=<span class="string">"vllm"</span>, model=<span class="string">"meta-llama/Llama-3-8b-chat-hf"</span>))

<span class="cm"># Any OpenAI-compatible endpoint</span>
provider = create_provider(LLMConfig(provider=<span class="string">"local"</span>, model=<span class="string">"model-name"</span>, base_url=<span class="string">"http://localhost:8080/v1"</span>))</code></pre>

<h2>Tier 3 — All Other Providers</h2>
<p>Each uses the same <code>create_provider(LLMConfig(provider="name", model="model"))</code> pattern:</p>
<table style="width:100%;font-size:12px;border-collapse:collapse;">
<tr style="background:var(--gray-100);"><th style="padding:8px;text-align:left;">Provider</th><th style="padding:8px;">Env Variable</th><th style="padding:8px;">Example Model</th></tr>
<tr><td style="padding:6px 8px;">cerebras</td><td>CEREBRAS_API_KEY</td><td>llama3.1-70b</td></tr>
<tr><td style="padding:6px 8px;">sambanova</td><td>SAMBANOVA_API_KEY</td><td>Meta-Llama-3.1-70B</td></tr>
<tr><td style="padding:6px 8px;">ai21</td><td>AI21_API_KEY</td><td>jamba-1.5-large</td></tr>
<tr><td style="padding:6px 8px;">replicate</td><td>REPLICATE_API_TOKEN</td><td>meta/llama-3-70b</td></tr>
<tr><td style="padding:6px 8px;">huggingface</td><td>HF_TOKEN</td><td>meta-llama/Llama-3-8b</td></tr>
<tr><td style="padding:6px 8px;">deepinfra</td><td>DEEPINFRA_API_KEY</td><td>meta-llama/Llama-3-70b</td></tr>
<tr><td style="padding:6px 8px;">azure_openai</td><td>AZURE_OPENAI_API_KEY</td><td>gpt-4o (deployment)</td></tr>
<tr><td style="padding:6px 8px;">moonshot</td><td>MOONSHOT_API_KEY</td><td>moonshot-v1-8k</td></tr>
<tr><td style="padding:6px 8px;">zhipu / glm</td><td>ZHIPU_API_KEY</td><td>glm-4</td></tr>
<tr><td style="padding:6px 8px;">qwen / dashscope</td><td>DASHSCOPE_API_KEY</td><td>qwen-max</td></tr>
<tr><td style="padding:6px 8px;">yi</td><td>YI_API_KEY</td><td>yi-large</td></tr>
<tr><td style="padding:6px 8px;">baichuan</td><td>BAICHUAN_API_KEY</td><td>Baichuan4</td></tr>
<tr><td style="padding:6px 8px;">minimax</td><td>MINIMAX_API_KEY</td><td>abab6.5-chat</td></tr>
<tr><td style="padding:6px 8px;">stepfun</td><td>STEPFUN_API_KEY</td><td>step-2-16k</td></tr>
<tr><td style="padding:6px 8px;">volcengine / doubao</td><td>VOLC_API_KEY / ARK_API_KEY</td><td>doubao-pro-32k</td></tr>
<tr><td style="padding:6px 8px;">hunyuan</td><td>HUNYUAN_API_KEY</td><td>hunyuan-pro</td></tr>
<tr><td style="padding:6px 8px;">spark</td><td>SPARK_API_KEY</td><td>spark-v4</td></tr>
<tr><td style="padding:6px 8px;">reka</td><td>REKA_API_KEY</td><td>reka-core</td></tr>
<tr><td style="padding:6px 8px;">upstage</td><td>UPSTAGE_API_KEY</td><td>solar-pro</td></tr>
<tr><td style="padding:6px 8px;">writer</td><td>WRITER_API_KEY</td><td>palmyra-x-004</td></tr>
<tr><td style="padding:6px 8px;">cloudflare</td><td>CLOUDFLARE_API_TOKEN</td><td>@cf/meta/llama-3-8b</td></tr>
<tr><td style="padding:6px 8px;">konko, friendli, novita, featherless, pipeshift, runpod, modal, baseten, lepton, nebius, anyscale, maritalk, aimlapi, edenai, predictionguard, gradient, oci_genai, watsonx, snowflake_cortex, databricks, sagemaker, dappier, greennode, aleph_alpha, netmind, abso</td><td colspan="2"><em>See env var naming convention: PROVIDER_API_KEY</em></td></tr>
</table>
</div>

<!-- ── Embeddings Detail ── -->
<div id="page-int-embed-detail" class="page">
<h1>Embeddings — 86 Providers</h1>
<p>Convert text to vectors for semantic search. All embedders share the same API: <code>embed(text)</code> and <code>embed_many(texts)</code>.</p>

<h2>Cloud API Embeddings</h2>

<h3>OpenAI Embeddings</h3>
<p>Industry standard. text-embedding-3-small (1536d) and text-embedding-3-large (3072d).</p>
<pre class="code-block"><code><span class="cm"># Set: OPENAI_API_KEY=sk-...</span>
<span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> OpenAIEmbedder

embedder = OpenAIEmbedder(model=<span class="string">"text-embedding-3-small"</span>)  <span class="cm"># 1536 dimensions</span>
vector = embedder.embed(<span class="string">"Hello world"</span>)          <span class="cm"># list[float], len=1536</span>
vectors = embedder.embed_many([<span class="string">"a"</span>, <span class="string">"b"</span>, <span class="string">"c"</span>])  <span class="cm"># Batched API call</span>
<span class="kw">print</span>(embedder.dimension)                       <span class="cm"># 1536</span></code></pre>

<h3>Cohere Embeddings</h3>
<p>High-quality multilingual embeddings with input_type control (search_document vs search_query).</p>
<pre class="code-block"><code><span class="cm"># Set: COHERE_API_KEY=...</span>
<span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> CohereEmbedder

embedder = CohereEmbedder(
    model=<span class="string">"embed-english-v3.0"</span>,    <span class="cm"># or "embed-multilingual-v3.0"</span>
    input_type=<span class="string">"search_document"</span>,  <span class="cm"># or "search_query" at query time</span>
)</code></pre>

<h3>Voyage AI Embeddings</h3>
<pre class="code-block"><code><span class="cm"># Set: VOYAGE_API_KEY=...</span>
<span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> VoyageEmbedder
embedder = VoyageEmbedder(model=<span class="string">"voyage-3"</span>)  <span class="cm"># 1024 dimensions</span></code></pre>

<h3>Google, NVIDIA, Jina, Nomic</h3>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> GoogleEmbedder, NVIDIAEmbedder, JinaEmbedder, NomicEmbedder

embedder = GoogleEmbedder(model=<span class="string">"models/text-embedding-004"</span>)     <span class="cm"># GOOGLE_API_KEY</span>
embedder = NVIDIAEmbedder(model=<span class="string">"nvidia/nv-embedqa-e5-v5"</span>)       <span class="cm"># NVIDIA_API_KEY</span>
embedder = JinaEmbedder(model=<span class="string">"jina-embeddings-v3"</span>)             <span class="cm"># JINA_API_KEY</span>
embedder = NomicEmbedder(model=<span class="string">"nomic-embed-text-v1.5"</span>)         <span class="cm"># NOMIC_API_KEY</span></code></pre>

<h2>Local Embeddings (Free, No API Key)</h2>

<h3>HuggingFace / Sentence Transformers</h3>
<p>Run any HuggingFace embedding model locally. Best quality-to-speed ratio.</p>
<pre class="code-block"><code><span class="cm"># pip install sentence-transformers</span>
<span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> HuggingFaceEmbedder

embedder = HuggingFaceEmbedder(<span class="string">"all-MiniLM-L6-v2"</span>)        <span class="cm"># 384d, fast</span>
embedder = HuggingFaceEmbedder(<span class="string">"BAAI/bge-large-en-v1.5"</span>)  <span class="cm"># 1024d, accurate</span>
embedder = HuggingFaceEmbedder(<span class="string">"intfloat/e5-large-v2"</span>)    <span class="cm"># 1024d, multilingual</span></code></pre>

<h3>FastEmbed (Qdrant) — Optimized ONNX</h3>
<pre class="code-block"><code><span class="cm"># pip install fastembed</span>
<span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> FastEmbedEmbedder
embedder = FastEmbedEmbedder(<span class="string">"BAAI/bge-small-en-v1.5"</span>)  <span class="cm"># ONNX optimized, very fast</span></code></pre>

<h3>Ollama, SpaCy, GPT4All, LlamaCpp</h3>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> OllamaEmbedder, SpacyEmbedder, GPT4AllEmbedder

embedder = OllamaEmbedder()                <span class="cm"># Uses Ollama server locally</span>
embedder = SpacyEmbedder(<span class="string">"en_core_web_md"</span>)  <span class="cm"># pip install spacy</span>
embedder = GPT4AllEmbedder()               <span class="cm"># pip install gpt4all</span></code></pre>

<h3>LocalEmbedder — Zero Dependencies</h3>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> LocalEmbedder
embedder = LocalEmbedder(dimension=384)  <span class="cm"># Hash-based pseudo-embeddings (testing only)</span></code></pre>

<h2>All 86 Embedders</h2>
<p>Cloud: OpenAI, Cohere, Voyage, Google, NVIDIA, Jina, Nomic, Azure, Bedrock, Mistral, Together, Fireworks, DeepInfra, Groq, OpenRouter, Anyscale, Perplexity, Lepton, Nebius, Databricks, Cloudflare, xAI, Snowflake, OCIGenAI, HunYuan, Doubao, StepFun, DashScope, VolcEngine, Spark, Zhipu, Baichuan, Qianfan, Gradient, MiniMax, Upstage, Solar, Watsonx, YandexGPT, Naver, MosaicML, BookendAI, Embaas, LLMRails, NLPCloud, Isaacus, Konko, OVHCloud, PremAI, EdenAI, PredictionGuard, Clarifai, Aleph Alpha</p>
<p>Local: HuggingFace, FastEmbed, Ollama, LMStudio, vLLM, SpaCy, GPT4All, LlamaCpp, OpenVINO, Model2Vec, LocalEmbedder, TEI, Infinity, SageMaker, LocalAI, IPEXLLM, LASER, Ascend, ModelScope, TextEmbed, TitanTakeoff, OpenClip, BGE, E5, GTE, Instructor, Mxbai, NomicLocal, SnowflakeArctic, Stella, MultilingualE5</p>
</div>

<!-- ── Vector Stores Detail ── -->
<div id="page-int-vs-detail" class="page">
<h1>Vector Stores — 153 Backends</h1>
<p>Store and search document embeddings. All vector stores share the same API: <code>add()</code>, <code>search()</code>, <code>delete()</code>, <code>count()</code>.</p>

<h2>Full Implementations (32)</h2>
<p>These have complete, production-ready implementations:</p>

<h3>In-Memory — Zero Dependencies</h3>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.vectorstore <span class="kw">import</span> InMemoryVectorStore
store = InMemoryVectorStore(embedder)
ids = store.add(documents)
results = store.search(<span class="string">"query"</span>, top_k=5)  <span class="cm"># Cosine similarity</span></code></pre>

<h3>FAISS — Fast ANN with Persistence</h3>
<pre class="code-block"><code><span class="cm"># pip install faiss-cpu  (or faiss-gpu)</span>
<span class="kw">from</span> duxx_ai.rag.vectorstore <span class="kw">import</span> FAISSVectorStore

store = FAISSVectorStore(embedder, dimension=1536)
store.add(documents)
results = store.search(<span class="string">"query"</span>, top_k=10)

<span class="cm"># Save/load index</span>
store.save(<span class="string">"my_index.faiss"</span>)
store = FAISSVectorStore.load(<span class="string">"my_index.faiss"</span>, embedder)</code></pre>

<h3>ChromaDB — Developer Friendly</h3>
<pre class="code-block"><code><span class="cm"># pip install chromadb</span>
<span class="kw">from</span> duxx_ai.rag.vectorstore <span class="kw">import</span> ChromaVectorStore

store = ChromaVectorStore(embedder,
    collection_name=<span class="string">"my_docs"</span>,
    persist_directory=<span class="string">"./chroma_data"</span>,  <span class="cm"># Persistent storage</span>
)</code></pre>

<h3>Pinecone — Managed Cloud</h3>
<pre class="code-block"><code><span class="cm"># pip install pinecone</span>
<span class="cm"># Set: PINECONE_API_KEY=...</span>
<span class="kw">from</span> duxx_ai.rag.vectorstore <span class="kw">import</span> PineconeVectorStore
store = PineconeVectorStore(embedder, index_name=<span class="string">"my-index"</span>)</code></pre>

<h3>Qdrant, Weaviate, Milvus, Elasticsearch, PGVector, Redis, MongoDB Atlas, LanceDB, OpenSearch, Neo4j, Cassandra, etc.</h3>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.vectorstore <span class="kw">import</span> (
    QdrantVectorStore,          <span class="cm"># pip install qdrant-client</span>
    WeaviateVectorStore,        <span class="cm"># pip install weaviate-client</span>
    MilvusVectorStore,          <span class="cm"># pip install pymilvus</span>
    ElasticsearchVectorStore,   <span class="cm"># pip install elasticsearch</span>
    PGVectorStore,              <span class="cm"># pip install psycopg2-binary pgvector</span>
    RedisVectorStore,           <span class="cm"># pip install redis</span>
    MongoDBAtlasVectorStore,    <span class="cm"># pip install pymongo</span>
    LanceDBVectorStore,         <span class="cm"># pip install lancedb</span>
    Neo4jVectorStore,           <span class="cm"># pip install neo4j</span>
    DuckDBVectorStore,          <span class="cm"># pip install duckdb</span>
)</code></pre>

<h2>Stub Implementations (121)</h2>
<p>These provide a clean import + fallback to InMemoryVectorStore when the native library isn't installed. Install the required package for full functionality.</p>
<p>Cloud: AstraDB, AlloyDB, BigQuery, CloudSQL, Spanner, Bigtable, Firestore, CosmosDB, AzureAISearch, DocumentDB, MemoryDB, Neptune, Snowflake, CockroachDB, Neon, Timescale, Supabase, Upstash, Turbopuffer, Vectara...</p>
<p>Self-hosted: Rockset, StarRocks, OceanBase, AnalyticDB, Hologres, DashVector, MyScale, FalkorDB, Vespa, Marqo, Meilisearch, ClickHouse, SingleStore, TiDB, MySQL...</p>
<p>ML: Annoy, ScaNN, USearch, HNSWLib, NMSLib, PyNNDescent, NGT, SKLearn...</p>
</div>

<!-- ── Doc Loaders Detail ── -->
<div id="page-int-loader-detail" class="page">
<h1>Document Loaders — 63 Types</h1>
<p>Load content from any source into Duxx AI <code>Document</code> objects. Each loader has a <code>.load()</code> method returning <code>list[Document]</code>.</p>

<h2>File Formats (14)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.loaders <span class="kw">import</span> *

docs = TextLoader(<span class="string">"readme.txt"</span>).load()
docs = CSVLoader(<span class="string">"data.csv"</span>).load()              <span class="cm"># One Document per row</span>
docs = JSONLoader(<span class="string">"data.json"</span>, content_key=<span class="string">"text"</span>).load()
docs = JSONLLoader(<span class="string">"data.jsonl"</span>).load()           <span class="cm"># One Document per line</span>
docs = MarkdownLoader(<span class="string">"docs.md"</span>, split_by_headers=<span class="kw">True</span>).load()  <span class="cm"># Split by ## headers</span>
docs = HTMLLoader(<span class="string">"page.html"</span>).load()             <span class="cm"># Strips tags, extracts title</span>
docs = PDFLoader(<span class="string">"paper.pdf"</span>).load()              <span class="cm"># pip install pdfplumber</span>
docs = DocxLoader(<span class="string">"report.docx"</span>).load()           <span class="cm"># pip install python-docx</span>
docs = ExcelLoader(<span class="string">"data.xlsx"</span>).load()            <span class="cm"># pip install openpyxl</span>
docs = PPTXLoader(<span class="string">"slides.pptx"</span>).load()           <span class="cm"># pip install python-pptx</span>
docs = EPUBLoader(<span class="string">"book.epub"</span>).load()             <span class="cm"># pip install ebooklib</span>
docs = RTFLoader(<span class="string">"doc.rtf"</span>).load()                <span class="cm"># pip install striprtf</span>
docs = XMLLoader(<span class="string">"data.xml"</span>, text_tags=[<span class="string">"title"</span>, <span class="string">"body"</span>]).load()
docs = EmailLoader(<span class="string">"msg.eml"</span>).load()              <span class="cm"># Extracts subject, from, body</span></code></pre>

<h2>Cloud Storage (6)</h2>
<pre class="code-block"><code>docs = S3Loader(<span class="string">"bucket"</span>, prefix=<span class="string">"docs/"</span>).load()               <span class="cm"># pip install boto3</span>
docs = GCSLoader(<span class="string">"bucket"</span>, prefix=<span class="string">"docs/"</span>).load()              <span class="cm"># pip install google-cloud-storage</span>
docs = AzureBlobLoader(<span class="string">"container"</span>).load()                    <span class="cm"># pip install azure-storage-blob</span>
docs = GoogleDriveLoader(file_ids=[<span class="string">"id1"</span>]).load()               <span class="cm"># GOOGLE_API_KEY</span>
docs = DropboxLoader(paths=[<span class="string">"/doc.txt"</span>]).load()                 <span class="cm"># DROPBOX_ACCESS_TOKEN</span>
docs = OneDriveLoader(file_ids=[<span class="string">"id1"</span>]).load()                  <span class="cm"># ONEDRIVE_ACCESS_TOKEN</span></code></pre>

<h2>Productivity Tools (9)</h2>
<pre class="code-block"><code>docs = NotionLoader(page_ids=[<span class="string">"page-id"</span>]).load()                <span class="cm"># NOTION_TOKEN</span>
docs = ConfluenceLoader(page_ids=[<span class="string">"id"</span>]).load()                  <span class="cm"># CONFLUENCE_TOKEN</span>
docs = JiraLoader(jql=<span class="string">"project=PROJ"</span>).load()                    <span class="cm"># JIRA_TOKEN</span>
docs = TrelloLoader(board_id=<span class="string">"id"</span>).load()                       <span class="cm"># TRELLO_API_KEY</span>
docs = LinearLoader(team_key=<span class="string">"TEAM"</span>).load()                     <span class="cm"># LINEAR_API_KEY</span>
docs = AirtableLoader(<span class="string">"base_id"</span>, <span class="string">"table"</span>).load()                <span class="cm"># AIRTABLE_API_KEY</span>
docs = AsanaLoader(project_gid=<span class="string">"id"</span>).load()                     <span class="cm"># ASANA_ACCESS_TOKEN</span>
docs = MondayLoader(board_id=<span class="string">"id"</span>).load()                       <span class="cm"># MONDAY_API_KEY</span>
docs = ClickUpLoader(list_id=<span class="string">"id"</span>).load()                       <span class="cm"># CLICKUP_API_KEY</span></code></pre>

<h2>Dev Tools, Messaging, Web, Academic, etc.</h2>
<pre class="code-block"><code><span class="cm"># Dev tools</span>
docs = GitHubLoader(<span class="string">"owner/repo"</span>).load()                        <span class="cm"># GITHUB_TOKEN</span>
docs = GitLabLoader(project_id=<span class="string">"123"</span>).load()                    <span class="cm"># GITLAB_TOKEN</span>

<span class="cm"># Messaging</span>
docs = SlackLoader(channel_id=<span class="string">"C123"</span>).load()                    <span class="cm"># SLACK_BOT_TOKEN</span>
docs = DiscordLoader(channel_id=<span class="string">"id"</span>).load()                    <span class="cm"># DISCORD_BOT_TOKEN</span>
docs = TelegramLoader(<span class="string">"export.json"</span>).load()                    <span class="cm"># JSON export file</span>
docs = WhatsAppLoader(<span class="string">"chat.txt"</span>).load()                       <span class="cm"># TXT export file</span>
docs = TwitterLoader(query=<span class="string">"AI agents"</span>).load()                  <span class="cm"># TWITTER_BEARER_TOKEN</span>

<span class="cm"># Web</span>
docs = WebLoader(<span class="string">"https://example.com"</span>).load()
docs = RecursiveURLLoader(<span class="string">"https://docs.example.com"</span>, max_depth=2).load()
docs = SitemapLoader(<span class="string">"https://example.com/sitemap.xml"</span>).load()
docs = FirecrawlLoader(<span class="string">"https://example.com"</span>).load()            <span class="cm"># FIRECRAWL_API_KEY</span>

<span class="cm"># Academic</span>
docs = WikipediaLoader(<span class="string">"machine learning"</span>).load()
docs = ArxivLoader(<span class="string">"transformer architecture"</span>).load()
docs = YouTubeLoader(<span class="string">"https://youtube.com/watch?v=..."</span>).load()   <span class="cm"># pip install youtube-transcript-api</span>

<span class="cm"># Social</span>
docs = RedditLoader(subreddit=<span class="string">"MachineLearning"</span>).load()
docs = HackerNewsLoader(limit=10).load()
docs = RSSLoader(<span class="string">"https://feed.example.com/rss"</span>).load()
docs = FigmaLoader(file_key=<span class="string">"key"</span>).load()                       <span class="cm"># FIGMA_ACCESS_TOKEN</span>

<span class="cm"># Batch</span>
docs = DirectoryLoader(<span class="string">"./data"</span>, glob=<span class="string">"**/*.md"</span>).load()          <span class="cm"># Auto-detects file types</span></code></pre>
</div>

<!-- ── Retrievers Detail ── -->
<div id="page-int-ret-detail" class="page">
<h1>Retrievers — 94 Types</h1>
<p>Retrieve relevant documents from any source. All retrievers share: <code>retrieve(query, top_k) -> list[Document]</code>.</p>

<h2>Core Retrieval Strategies (13)</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.retriever <span class="kw">import</span> *

<span class="cm"># Vector similarity</span>
ret = VectorRetriever(store, min_score=0.5)

<span class="cm"># BM25 (Okapi) — probabilistic keyword ranking</span>
ret = BM25Retriever(documents, k1=1.5, b=0.75)

<span class="cm"># Hybrid — vector + keyword with Reciprocal Rank Fusion</span>
ret = HybridRetriever(vector_ret, keyword_ret, vector_weight=0.6)

<span class="cm"># Multi-query — generates query variations for better recall</span>
ret = MultiQueryRetriever(base_ret, query_count=3)

<span class="cm"># Reranker — Cohere or local CrossEncoder</span>
ret = RerankerRetriever(base_ret, method=<span class="string">"cohere"</span>, fetch_k=20)
ret = RerankerRetriever(base_ret, method=<span class="string">"cross_encoder"</span>)

<span class="cm"># Ensemble — combine any retrievers with weights</span>
ret = EnsembleRetriever([vec_ret, bm25_ret, wiki_ret], weights=[0.5, 0.3, 0.2])

<span class="cm"># Parent document — match chunks, return full parents</span>
ret = ParentDocRetriever(chunk_retriever, parent_documents)

<span class="cm"># Contextual compression — keep only relevant sentences</span>
ret = ContextualCompressionRetriever(base_ret, min_sentences=2)

<span class="cm"># MMR — balance relevance with diversity</span>
ret = MaxMarginalRelevanceRetriever(base_ret, lambda_mult=0.5)

<span class="cm"># Time-weighted — boost newer documents</span>
ret = TimeWeightedRetriever(base_ret, decay_rate=0.01)

<span class="cm"># Long context reorder — "Lost in the Middle" fix</span>
ret = LongContextReorderRetriever(base_ret)

<span class="cm"># Deduplication — remove near-duplicates</span>
ret = DeduplicationRetriever(base_ret, similarity_threshold=0.85)</code></pre>

<h2>Search API Retrievers (11)</h2>
<pre class="code-block"><code>ret = TavilyRetriever()          <span class="cm"># TAVILY_API_KEY — best for RAG</span>
ret = ExaRetriever()             <span class="cm"># EXA_API_KEY — semantic search</span>
ret = GoogleSearchRetriever()    <span class="cm"># GOOGLE_API_KEY + GOOGLE_CSE_ID</span>
ret = BingSearchRetriever()      <span class="cm"># BING_SEARCH_KEY</span>
ret = BraveSearchRetriever()     <span class="cm"># BRAVE_API_KEY</span>
ret = SerpAPIRetriever()         <span class="cm"># SERPAPI_API_KEY</span>
ret = SerperRetriever()          <span class="cm"># SERPER_API_KEY</span>
ret = YouRetriever()             <span class="cm"># YDC_API_KEY</span>
ret = SearchAPIRetriever()       <span class="cm"># SEARCHAPI_KEY</span>
ret = WebSearchRetriever()       <span class="cm"># DuckDuckGo (no API key)</span>
ret = SearxNGRetriever(url=<span class="string">"http://localhost:8888"</span>)  <span class="cm"># Self-hosted</span></code></pre>

<h2>Academic &amp; Knowledge (3)</h2>
<pre class="code-block"><code>ret = WikipediaRetriever(lang=<span class="string">"en"</span>)       <span class="cm"># No API key</span>
ret = ArxivRetriever(max_results=5)       <span class="cm"># No API key</span>
ret = PubMedRetriever()                   <span class="cm"># No API key</span></code></pre>

<h2>Cloud Managed (8)</h2>
<pre class="code-block"><code>ret = AmazonKendraRetriever(index_id=<span class="string">"id"</span>)
ret = AzureAISearchRetriever(index_name=<span class="string">"idx"</span>)
ret = ElasticsearchRetriever(index_name=<span class="string">"idx"</span>)
ret = CohereRerankRetriever(base_ret, model=<span class="string">"rerank-english-v3.0"</span>)
ret = VectaraRetriever(corpus_key=<span class="string">"key"</span>)
ret = VertexAISearchRetriever(data_store_id=<span class="string">"id"</span>)
ret = PineconeHybridRetriever(index_name=<span class="string">"idx"</span>, embedder=embedder)</code></pre>

<h2>ML-Based (4) &amp; Advanced Strategies (12+)</h2>
<pre class="code-block"><code><span class="cm"># ML-based (pip install scikit-learn)</span>
ret = KNNRetriever(documents, embedder=embedder)
ret = SVMRetriever(documents)
ret = TFIDFRetriever(documents)
ret = FlashRankRetriever(base_ret)   <span class="cm"># pip install flashrank</span>

<span class="cm"># Advanced strategy stubs</span>
ret = HyDERetriever()        <span class="cm"># Hypothetical Document Embeddings</span>
ret = RAGFusionRetriever()   <span class="cm"># RAG-Fusion multi-query + RRF</span>
ret = StepBackRetriever()    <span class="cm"># Step-back prompting</span>
ret = ColBERTRetriever()     <span class="cm"># Late interaction (pip install colbert-ai)</span>
ret = SPLADERetriever()      <span class="cm"># Sparse retriever (pip install splade)</span>
ret = RAGatouillRetriever()  <span class="cm"># RAGatouille ColBERT (pip install ragatouille)</span></code></pre>
</div>

<!-- ── Checkpointers Detail ── -->
<div id="page-int-check-detail" class="page">
<h1>Checkpointers — 8 Backends</h1>
<p>Persist FlowGraph state for recovery, time-travel, and durable execution. All checkpointers share: <code>save()</code>, <code>load()</code>, <code>list()</code>, <code>get_latest()</code>.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.orchestration.state_graph <span class="kw">import</span> *

<span class="cm"># In-memory (default, no deps)</span>
cp = MemorySnapshotStore()

<span class="cm"># SQLite (persistent, no server needed)</span>
cp = SQLiteSnapshotStore(<span class="string">"checkpoints.db"</span>)

<span class="cm"># File-based JSON</span>
cp = FileSnapshotStore(directory=<span class="string">".duxx_checkpoints"</span>)

<span class="cm"># PostgreSQL (pip install psycopg2-binary)</span>
cp = PostgresSnapshotStore(<span class="string">"postgresql://user:pass@host/db"</span>)

<span class="cm"># Redis (pip install redis)</span>
cp = RedisSnapshotStore(<span class="string">"redis://localhost:6379"</span>)

<span class="cm"># MongoDB (pip install pymongo)</span>
cp = MongoSnapshotStore(<span class="string">"mongodb://localhost:27017"</span>)

<span class="cm"># AWS DynamoDB (pip install boto3)</span>
cp = DynamoDBSnapshotStore(<span class="string">"table-name"</span>, region=<span class="string">"us-east-1"</span>)

<span class="cm"># Valkey/Redis fork (pip install redis)</span>
cp = ValleySnapshotStore(<span class="string">"redis://localhost:6379"</span>)

<span class="cm"># Use with FlowGraph</span>
compiled = graph.compile(checkpointer=cp)
result = <span class="kw">await</span> compiled.invoke({<span class="string">"input"</span>: <span class="string">"data"</span>})

<span class="cm"># Time-travel</span>
history = <span class="kw">await</span> compiled.get_state_history()
result = <span class="kw">await</span> compiled.replay_from(history[0].snapshot_id)</code></pre>
</div>

<!-- ── Output Parsers Detail ── -->
<div id="page-int-parser-detail" class="page">
<h1>Output Parsers — 10 Types</h1>
<p>Parse LLM text output into structured data. Each parser has <code>parse(text)</code> and <code>get_format_instructions()</code>.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.parsers <span class="kw">import</span> *

<span class="cm"># JSON — extract from code fences or raw text</span>
p = JSONOutputParser()
result = p.parse(<span class="string">'```json\n{"name": "Alice"}\n```'</span>)  <span class="cm"># {"name": "Alice"}</span>

<span class="cm"># Pydantic — validate against schema</span>
<span class="kw">class</span> User(BaseModel): name: <span class="kw">str</span>; age: <span class="kw">int</span>
p = PydanticOutputParser(User)
user = p.parse(<span class="string">'{"name": "Alice", "age": 30}'</span>)     <span class="cm"># User(name="Alice", age=30)</span>

<span class="cm"># XML — extract named tags</span>
p = XMLOutputParser(tags=[<span class="string">"name"</span>, <span class="string">"score"</span>])
result = p.parse(<span class="string">"&lt;name&gt;Alice&lt;/name&gt;&lt;score&gt;95&lt;/score&gt;"</span>)

<span class="cm"># YAML — parse YAML blocks</span>
p = YAMLOutputParser()
result = p.parse(<span class="string">"name: Alice\nage: 30"</span>)             <span class="cm"># {"name": "Alice", "age": 30}</span>

<span class="cm"># CSV — parse tabular data</span>
p = CSVOutputParser()
result = p.parse(<span class="string">"name,age\nAlice,30\nBob,25"</span>)      <span class="cm"># [{"name":"Alice","age":"30"}, ...]</span>

<span class="cm"># Enum — constrained to allowed values</span>
p = EnumOutputParser(choices=[<span class="string">"positive"</span>, <span class="string">"negative"</span>, <span class="string">"neutral"</span>])
result = p.parse(<span class="string">"The sentiment is positive"</span>)       <span class="cm"># "positive"</span>

<span class="cm"># Markdown — split by ## headers</span>
p = MarkdownOutputParser()
<span class="cm"># Regex — named capture groups</span>
p = RegexOutputParser(pattern=<span class="string">r"Name: (?P&lt;name&gt;.+)"</span>)
<span class="cm"># List — numbered/bulleted items</span>
p = ListOutputParser()
<span class="cm"># Retry — auto-retry on failure with error feedback</span>
p = RetryParser(JSONOutputParser(), max_retries=3)

<span class="cm"># Get format instructions for prompt</span>
instructions = p.get_format_instructions()
prompt = f<span class="string">"Answer the question.\n{instructions}\nQ: What is 2+2?"</span></code></pre>
</div>

<!-- ── Middleware Detail ── -->
<div id="page-int-mw-detail" class="page">
<h1>Middleware — 4 Types</h1>
<p>Wrap LLM calls with cross-cutting concerns. Chain multiple middleware together.</p>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.core.middleware <span class="kw">import</span> *

<span class="cm"># 1. Prompt Cache — avoid redundant LLM calls</span>
cache = PromptCacheMiddleware(ttl_seconds=300, max_entries=1000)
<span class="kw">print</span>(cache.stats)  <span class="cm"># {"hits": 42, "misses": 58, "size": 58}</span>

<span class="cm"># 2. Content Moderation — block PII, injection, profanity</span>
mod = ContentModerationMiddleware(
    block_pii=<span class="kw">True</span>,           <span class="cm"># SSN, credit card, email, phone</span>
    block_injection=<span class="kw">True</span>,     <span class="cm"># "ignore previous instructions"</span>
    block_profanity=<span class="kw">False</span>,    <span class="cm"># Optional</span>
    action=<span class="string">"redact"</span>,           <span class="cm"># "block" | "redact" | "warn"</span>
    custom_blocked_patterns=[<span class="string">r"secret.*key"</span>],  <span class="cm"># Custom regex</span>
)

<span class="cm"># 3. Logging — track all LLM calls</span>
log = LoggingMiddleware(log_prompts=<span class="kw">True</span>, log_responses=<span class="kw">False</span>)

<span class="cm"># 4. Rate Limiting — prevent API overuse</span>
rate = RateLimitMiddleware(max_calls_per_minute=60)

<span class="cm"># Chain them together</span>
chain = MiddlewareChain([rate, cache, mod, log])
metadata = <span class="kw">await</span> chain.before_call(conversation, system_prompt, {})
<span class="kw">if not</span> metadata.get(<span class="string">"_blocked"</span>):
    response = <span class="kw">await</span> provider.complete(conversation)
    response_text = <span class="kw">await</span> chain.after_call(response.content, metadata)</code></pre>
</div>

<!-- ── Contextual Retrieval Detail ── -->
<div id="page-int-contextual" class="page">
<h1>Contextual Retrieval</h1>
<p>Anthropic's technique that reduces retrieval failures by <strong>49-67%</strong>. Prepends AI-generated context to each chunk before embedding.</p>

<h2>How It Works</h2>
<p>Standard RAG splits documents into chunks, but chunks lose context. "The company's revenue grew by 3%" — which company? What period?</p>
<p>Contextual Retrieval uses an LLM to generate a short explanation for each chunk within the full document, then prepends it before embedding and indexing.</p>

<h3>Before (standard)</h3>
<pre class="code-block"><code><span class="cm">"The company's revenue grew by 3% over the previous quarter."</span></code></pre>

<h3>After (contextualized)</h3>
<pre class="code-block"><code><span class="cm">"This chunk is from ACME Corp's Q2 2023 SEC filing.</span>
<span class="cm">Previous quarter revenue was $314M.</span>
<span class="cm"></span>
<span class="cm">The company's revenue grew by 3% over the previous quarter."</span></code></pre>

<h2>Full Pipeline</h2>
<pre class="code-block"><code><span class="kw">from</span> duxx_ai.rag.contextual <span class="kw">import</span> ContextualRetrieval
<span class="kw">from</span> duxx_ai.rag.embeddings <span class="kw">import</span> OpenAIEmbedder
<span class="kw">from</span> duxx_ai.core.llm <span class="kw">import</span> create_provider, LLMConfig

embedder = OpenAIEmbedder()
llm = create_provider(LLMConfig(provider=<span class="string">"anthropic"</span>, model=<span class="string">"claude-haiku"</span>))

cr = ContextualRetrieval(
    embedder=embedder,
    llm_provider=llm,
    chunk_size=800,            <span class="cm"># Words per chunk</span>
    chunk_overlap=200,         <span class="cm"># Overlap between chunks</span>
    use_contextual_bm25=<span class="kw">True</span>, <span class="cm"># Enable contextual BM25</span>
    use_reranking=<span class="kw">True</span>,        <span class="cm"># Cohere reranking</span>
    initial_fetch_k=150,       <span class="cm"># Fetch 150 candidates</span>
    final_top_k=20,            <span class="cm"># Return top 20 after rerank</span>
)

<span class="cm"># Index documents</span>
<span class="kw">await</span> cr.add_document(full_text, source=<span class="string">"report.pdf"</span>)

<span class="cm"># Query</span>
results = <span class="kw">await</span> cr.query(<span class="string">"What was Q4 revenue growth?"</span>, top_k=5)
<span class="kw">for</span> doc <span class="kw">in</span> results:
    <span class="kw">print</span>(doc.content)

<span class="cm"># Stats</span>
<span class="kw">print</span>(cr.stats)
<span class="cm"># {"documents_indexed": 3, "total_chunks": 45, "context_cache_size": 45}</span></code></pre>

<h2>Without LLM (Heuristic Fallback)</h2>
<pre class="code-block"><code><span class="cm"># No API key needed — uses surrounding text as context</span>
cr = ContextualRetrieval(
    embedder=embedder,
    llm_provider=<span class="kw">None</span>,    <span class="cm"># Heuristic context generation</span>
)</code></pre>

<h2>Performance</h2>
<table style="width:100%;font-size:13px;border-collapse:collapse;">
<tr style="background:var(--gray-100);"><th style="padding:8px;text-align:left;">Method</th><th style="padding:8px;">Failure Reduction</th></tr>
<tr><td style="padding:6px 8px;">Contextual Embeddings only</td><td><strong>35%</strong></td></tr>
<tr><td style="padding:6px 8px;">+ Contextual BM25</td><td><strong>49%</strong></td></tr>
<tr><td style="padding:6px 8px;">+ Reranking</td><td><strong>67%</strong></td></tr>
</table>
</div>

</main>

<!-- ── TOC (placeholder, populated by JS) ── -->
<aside class="docs-toc" id="toc-panel">
    <h4>On This Page</h4>
</aside>

</div>

<script>
// ── Hero Particles ──
(function initParticles() {
    const container = document.getElementById('hero-particles');
    if (!container) return;
    for (let i = 0; i < 20; i++) {
        const p = document.createElement('div');
        p.className = 'hero-particle';
        p.style.left = Math.random() * 100 + '%';
        p.style.top = Math.random() * 100 + '%';
        p.style.setProperty('--dx', (Math.random() * 160 - 80) + 'px');
        p.style.setProperty('--dy', (Math.random() * -200 - 40) + 'px');
        p.style.setProperty('--duration', (6 + Math.random() * 10) + 's');
        p.style.setProperty('--delay', (Math.random() * 8) + 's');
        p.style.width = (2 + Math.random() * 4) + 'px';
        p.style.height = p.style.width;
        p.style.background = Math.random() > 0.5 ? 'rgba(255,255,255,0.25)' : 'rgba(214,51,108,0.3)';
        container.appendChild(p);
    }
})();

// ── SPA Navigation ──
function navigateTo(pageId) {
    document.querySelectorAll('.page').forEach(p => {
        p.classList.remove('active');
        p.style.animation = '';
    });
    document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));

    const page = document.getElementById('page-' + pageId);
    if (page) {
        page.classList.add('active');
        page.style.animation = 'fade-in-up 0.35s ease both';
        window.scrollTo(0, 0);
    }

    // Activate sidebar link
    document.querySelectorAll('.sidebar-link').forEach(l => {
        if (l.getAttribute('onclick') && l.getAttribute('onclick').includes(pageId)) {
            l.classList.add('active');
        }
    });

    // Update TOC
    updateTOC(pageId);

    // Update URL
    history.pushState(null, '', '/docs#' + pageId);
}

function updateTOC(pageId) {
    const page = document.getElementById('page-' + pageId);
    const toc = document.getElementById('toc-panel');
    if (!page || !toc) return;

    const headings = page.querySelectorAll('h2[id], h3[id]');
    let html = '<h4>On This Page</h4>';
    headings.forEach(h => {
        html += '<a href="#' + h.id + '" onclick="event.preventDefault(); document.getElementById(\\'' + h.id + '\\').scrollIntoView({behavior:\\'smooth\\'})">' + h.textContent + '</a>';
    });
    toc.innerHTML = html;
}

// Handle initial hash
window.addEventListener('DOMContentLoaded', () => {
    const hash = window.location.hash.replace('#', '');
    if (hash) navigateTo(hash);
    else updateTOC('overview');
});

window.addEventListener('hashchange', () => {
    const hash = window.location.hash.replace('#', '');
    if (hash) navigateTo(hash);
});
</script>

</body>
</html>"""
