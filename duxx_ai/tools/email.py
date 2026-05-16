"""Email domain tools for Duxx AI agents.

Provides tools for sending, reading, searching, and replying to emails.
These are placeholder implementations -- real usage requires SMTP/IMAP
configuration (host, port, credentials) to be injected via environment
variables or a config object.

Required config for production use:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD   (sending)
    IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASSWORD   (reading)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from duxx_ai.core.tool import Tool, tool

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="send_email",
    description="Send an email message. Requires SMTP configuration.",
    requires_approval=True,
    tags=["email", "communication"],
)
def send_email(to: str, subject: str, body: str, cc: str = "") -> str:
    """Send an email to one or more recipients.

    Args:
        to: Comma-separated recipient email addresses.
        subject: Email subject line.
        body: Plain-text email body.
        cc: Comma-separated CC addresses (optional).

    Returns:
        Confirmation string or error message.
    """
    # Validate inputs
    if not to or not to.strip():
        return "Error: 'to' address is required."
    if not subject or not subject.strip():
        return "Error: 'subject' is required."

    recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
    cc_list = [addr.strip() for addr in cc.split(",") if addr.strip()] if cc else []

    # Placeholder -- in production, use smtplib here.
    return (
        f"[PLACEHOLDER] Email queued for delivery.\n"
        f"  To: {', '.join(recipients)}\n"
        f"  CC: {', '.join(cc_list) if cc_list else '(none)'}\n"
        f"  Subject: {subject}\n"
        f"  Body length: {len(body)} chars\n"
        f"  Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"\n"
        f"Note: Configure SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD "
        f"environment variables for actual delivery."
    )


@tool(
    name="read_inbox",
    description="Read emails from a mailbox folder.",
    tags=["email", "communication"],
)
def read_inbox(folder: str = "inbox", limit: int = 10) -> str:
    """Read emails from the specified folder.

    Args:
        folder: Mailbox folder name (e.g. inbox, sent, drafts).
        limit: Maximum number of emails to return.

    Returns:
        JSON-formatted list of email summaries or error message.
    """
    if limit < 1:
        return "Error: limit must be >= 1."

    # Placeholder -- in production, use imaplib here.
    sample_emails = [
        {
            "id": f"msg_{i}",
            "from": f"sender{i}@example.com",
            "subject": f"Sample email #{i}",
            "date": datetime.now(timezone.utc).isoformat(),
            "snippet": f"This is a placeholder email body for message {i}...",
            "read": i % 2 == 0,
        }
        for i in range(1, min(limit, 5) + 1)
    ]

    return (
        f"[PLACEHOLDER] Showing {len(sample_emails)} emails from '{folder}'.\n"
        f"Configure IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASSWORD for real access.\n\n"
        + json.dumps(sample_emails, indent=2)
    )


@tool(
    name="search_email",
    description="Search emails by query string across folders.",
    tags=["email", "communication"],
)
def search_email(query: str, folder: str = "all", limit: int = 10) -> str:
    """Search for emails matching a query.

    Args:
        query: Search query (subject, sender, body keywords).
        folder: Folder to search in, or 'all' for all folders.
        limit: Maximum results to return.

    Returns:
        JSON-formatted list of matching email summaries.
    """
    if not query or not query.strip():
        return "Error: search query is required."
    if limit < 1:
        return "Error: limit must be >= 1."

    # Placeholder
    results = [
        {
            "id": f"msg_search_{i}",
            "from": f"match{i}@example.com",
            "subject": f"Re: {query} (result #{i})",
            "date": datetime.now(timezone.utc).isoformat(),
            "snippet": f"...matched '{query}' in email body...",
            "folder": folder if folder != "all" else "inbox",
        }
        for i in range(1, min(limit, 3) + 1)
    ]

    return (
        f"[PLACEHOLDER] Found {len(results)} results for '{query}' in '{folder}'.\n"
        f"Configure IMAP credentials for real search.\n\n"
        + json.dumps(results, indent=2)
    )


@tool(
    name="reply_email",
    description="Reply to an existing email by message ID.",
    requires_approval=True,
    tags=["email", "communication"],
)
def reply_email(message_id: str, body: str) -> str:
    """Reply to an email identified by its message ID.

    Args:
        message_id: The ID of the message to reply to.
        body: The reply body text.

    Returns:
        Confirmation string or error message.
    """
    if not message_id or not message_id.strip():
        return "Error: message_id is required."
    if not body or not body.strip():
        return "Error: reply body is required."

    return (
        f"[PLACEHOLDER] Reply queued.\n"
        f"  In-Reply-To: {message_id}\n"
        f"  Body length: {len(body)} chars\n"
        f"  Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"\n"
        f"Note: Configure SMTP and IMAP credentials for actual delivery."
    )


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_TOOLS: dict[str, Tool] = {
    "send_email": send_email,
    "read_inbox": read_inbox,
    "search_email": search_email,
    "reply_email": reply_email,
}


def get_email_tools(names: list[str] | None = None) -> list[Tool]:
    """Get email tools by name. If names is None, return all."""
    if names is None:
        return list(MODULE_TOOLS.values())
    return [MODULE_TOOLS[n] for n in names if n in MODULE_TOOLS]


try:
    from duxx_ai.tools.registry import register_domain

    register_domain("email", MODULE_TOOLS)
except ImportError:
    pass
