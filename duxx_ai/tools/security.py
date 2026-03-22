"""Security domain tools for Duxx AI agents.

Provides tools for vulnerability scanning, compliance checking, log
analysis, and SSL certificate inspection. check_ssl performs real
SSL/TLS inspection using the ssl and socket stdlib modules.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from duxx_ai.core.tool import Tool, tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="scan_vulnerabilities",
    description="Scan a target for common security vulnerabilities.",
    requires_approval=True,
    tags=["security", "scanning"],
)
def scan_vulnerabilities(target: str, scan_type: str = "basic") -> str:
    """Run a security vulnerability scan against a target.

    Args:
        target: Target to scan -- URL, IP address, or hostname.
        scan_type: Scan depth -- 'basic', 'standard', or 'deep'.

    Returns:
        JSON-formatted scan results (placeholder).
    """
    if not target or not target.strip():
        return "Error: target is required."

    valid_types = {"basic", "standard", "deep"}
    if scan_type not in valid_types:
        return f"Error: invalid scan_type '{scan_type}'. Use one of: {', '.join(sorted(valid_types))}"

    # Placeholder -- in production, integrate with a scanner like nmap, Trivy, etc.
    sample_findings = [
        {
            "severity": "high",
            "title": "Outdated TLS version detected",
            "description": "Target supports TLS 1.0 which is deprecated.",
            "recommendation": "Disable TLS 1.0 and 1.1; enforce TLS 1.2+.",
            "cve": "N/A",
        },
        {
            "severity": "medium",
            "title": "Missing security headers",
            "description": "X-Content-Type-Options and X-Frame-Options headers not set.",
            "recommendation": "Add X-Content-Type-Options: nosniff and X-Frame-Options: DENY.",
            "cve": "N/A",
        },
        {
            "severity": "low",
            "title": "Server version disclosure",
            "description": "Server header reveals software version.",
            "recommendation": "Remove or obfuscate the Server header.",
            "cve": "N/A",
        },
    ]

    # Filter findings by scan depth
    if scan_type == "basic":
        findings = sample_findings[:1]
    elif scan_type == "standard":
        findings = sample_findings[:2]
    else:
        findings = sample_findings

    result = {
        "target": target,
        "scan_type": scan_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "findings_count": len(findings),
        "findings": findings,
    }

    return (
        f"[PLACEHOLDER] Vulnerability scan completed.\n"
        f"Integrate with nmap, Trivy, or OWASP ZAP for real scanning.\n\n"
        + json.dumps(result, indent=2)
    )


@tool(
    name="check_compliance",
    description="Check a configuration against a compliance framework.",
    tags=["security", "compliance"],
)
def check_compliance(config: str, framework: str = "SOC2") -> str:
    """Evaluate a configuration against a compliance framework.

    Args:
        config: JSON-encoded configuration to check.
        framework: Compliance framework -- SOC2, HIPAA, PCI_DSS, ISO27001, GDPR.

    Returns:
        JSON compliance report.
    """
    if not config or not config.strip():
        return "Error: config is required."

    valid_frameworks = {"SOC2", "HIPAA", "PCI_DSS", "ISO27001", "GDPR"}
    framework = framework.upper().replace("-", "_")
    if framework not in valid_frameworks:
        return (
            f"Error: unsupported framework '{framework}'. "
            f"Use one of: {', '.join(sorted(valid_frameworks))}"
        )

    try:
        config_dict = json.loads(config)
    except json.JSONDecodeError as e:
        return f"Error: invalid config JSON -- {e}"

    # Placeholder compliance checks
    checks = [
        {
            "control": f"{framework}-001",
            "name": "Encryption at rest",
            "status": "pass" if config_dict.get("encryption_at_rest") else "fail",
            "details": "Data encryption at rest must be enabled.",
        },
        {
            "control": f"{framework}-002",
            "name": "Access logging",
            "status": "pass" if config_dict.get("access_logging") else "fail",
            "details": "Access logging must be enabled for audit trails.",
        },
        {
            "control": f"{framework}-003",
            "name": "MFA enforcement",
            "status": "pass" if config_dict.get("mfa_enabled") else "fail",
            "details": "Multi-factor authentication must be enforced.",
        },
        {
            "control": f"{framework}-004",
            "name": "Password policy",
            "status": "pass" if config_dict.get("password_min_length", 0) >= 12 else "fail",
            "details": "Passwords must be at least 12 characters.",
        },
    ]

    passed = sum(1 for c in checks if c["status"] == "pass")
    total = len(checks)

    report = {
        "framework": framework,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_score": f"{passed}/{total}",
        "compliant": passed == total,
        "checks": checks,
    }

    return json.dumps(report, indent=2)


@tool(
    name="analyze_log",
    description="Analyze log content for security patterns and anomalies.",
    tags=["security", "logging"],
)
def analyze_log(log_content: str, pattern: str = "anomaly") -> str:
    """Analyze log text for security-relevant patterns.

    Args:
        log_content: The raw log text to analyze.
        pattern: Analysis mode -- 'anomaly', 'auth_failure', 'injection',
            'brute_force', or a custom regex pattern.

    Returns:
        JSON-formatted analysis results.
    """
    if not log_content or not log_content.strip():
        return "Error: log_content is required."

    lines = log_content.strip().splitlines()
    findings: list[dict[str, Any]] = []

    # Built-in pattern definitions
    patterns = {
        "anomaly": [
            (r"(?i)(error|fail|exception|critical|fatal)", "error_keyword"),
            (r"(?i)(denied|forbidden|unauthorized|403|401)", "access_denied"),
            (r"(?i)(timeout|timed?\s*out)", "timeout"),
        ],
        "auth_failure": [
            (r"(?i)(login\s+fail|authentication\s+fail|invalid\s+(password|credential))", "auth_fail"),
            (r"(?i)(401|unauthorized)", "http_401"),
            (r"(?i)(locked\s+out|account\s+locked)", "account_locked"),
        ],
        "injection": [
            (r"(?i)(union\s+select|drop\s+table|;\s*delete|;\s*insert)", "sql_injection"),
            (r"(?i)(<script|javascript:|on\w+\s*=)", "xss_attempt"),
            (r"(?i)(\.\./|\.\.\\|%2e%2e)", "path_traversal"),
        ],
        "brute_force": [
            (r"(?i)(failed\s+login|bad\s+password|invalid\s+credentials)", "failed_login"),
            (r"(?i)(rate\s+limit|too\s+many\s+requests|429)", "rate_limited"),
        ],
    }

    # Get patterns to use
    if pattern in patterns:
        regex_list = patterns[pattern]
    else:
        # Treat as custom regex
        try:
            re.compile(pattern)
            regex_list = [(pattern, "custom_match")]
        except re.error as e:
            return f"Error: invalid regex pattern -- {e}"

    for line_num, line in enumerate(lines, start=1):
        for regex, category in regex_list:
            match = re.search(regex, line)
            if match:
                findings.append({
                    "line_number": line_num,
                    "category": category,
                    "matched_text": match.group(0),
                    "line": line.strip()[:200],
                })

    result = {
        "pattern": pattern,
        "total_lines": len(lines),
        "findings_count": len(findings),
        "findings": findings[:100],  # cap at 100 findings
    }

    return json.dumps(result, indent=2)


@tool(
    name="check_ssl",
    description="Check SSL/TLS certificate details for a domain.",
    tags=["security", "ssl", "tls"],
)
def check_ssl(domain: str) -> str:
    """Inspect the SSL/TLS certificate of a domain.

    Connects to the domain on port 443 and retrieves certificate details
    including issuer, validity dates, SANs, and protocol version.

    Args:
        domain: Domain name to check (e.g. example.com).

    Returns:
        JSON-formatted certificate details.
    """
    import socket
    import ssl

    if not domain or not domain.strip():
        return "Error: domain is required."

    # Strip protocol prefix if provided
    domain = domain.strip()
    for prefix in ("https://", "http://"):
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
    domain = domain.rstrip("/").split("/")[0]  # remove path
    domain = domain.split(":")[0]  # remove port

    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                protocol = ssock.version()
                cipher = ssock.cipher()

        if not cert:
            return f"Error: no certificate returned for {domain}."

        # Parse certificate fields
        subject = dict(x[0] for x in cert.get("subject", ()))
        issuer = dict(x[0] for x in cert.get("issuer", ()))

        # Parse dates
        not_before = cert.get("notBefore", "")
        not_after = cert.get("notAfter", "")

        # Calculate days until expiry
        days_until_expiry = None
        if not_after:
            try:
                expiry = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
                days_until_expiry = (expiry - datetime.utcnow()).days
            except ValueError:
                pass

        # Subject Alternative Names
        san_entries = []
        for san_type, san_value in cert.get("subjectAltName", ()):
            san_entries.append({"type": san_type, "value": san_value})

        result = {
            "domain": domain,
            "subject": subject,
            "issuer": issuer,
            "serial_number": cert.get("serialNumber"),
            "not_before": not_before,
            "not_after": not_after,
            "days_until_expiry": days_until_expiry,
            "expired": days_until_expiry is not None and days_until_expiry < 0,
            "protocol": protocol,
            "cipher": {
                "name": cipher[0] if cipher else None,
                "protocol": cipher[1] if cipher and len(cipher) > 1 else None,
                "bits": cipher[2] if cipher and len(cipher) > 2 else None,
            },
            "san": san_entries,
        }

        return json.dumps(result, indent=2)

    except socket.gaierror:
        return f"Error: could not resolve domain '{domain}'."
    except socket.timeout:
        return f"Error: connection to {domain}:443 timed out."
    except ssl.SSLCertVerificationError as e:
        return f"SSL verification error for {domain}: {e}"
    except ConnectionRefusedError:
        return f"Error: connection refused to {domain}:443."
    except Exception as e:
        return f"Error checking SSL for {domain}: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_TOOLS: dict[str, Tool] = {
    "scan_vulnerabilities": scan_vulnerabilities,
    "check_compliance": check_compliance,
    "analyze_log": analyze_log,
    "check_ssl": check_ssl,
}


def get_security_tools(names: list[str] | None = None) -> list[Tool]:
    """Get security tools by name. If names is None, return all."""
    if names is None:
        return list(MODULE_TOOLS.values())
    return [MODULE_TOOLS[n] for n in names if n in MODULE_TOOLS]


try:
    from duxx_ai.tools.registry import register_domain

    register_domain("security", MODULE_TOOLS)
except ImportError:
    pass
