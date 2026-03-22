"""Calendar domain tools for Duxx AI agents.

Provides tools for scheduling meetings, checking availability, listing
events, and cancelling events. These are placeholder implementations --
real usage requires integration with a calendar API (Google Calendar,
Microsoft Graph, CalDAV, etc.).

Required config for production use:
    CALENDAR_PROVIDER   (google, microsoft, caldav)
    CALENDAR_CREDENTIALS_PATH or CALENDAR_API_KEY
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from duxx_ai.core.tool import Tool, tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="schedule_meeting",
    description="Schedule a new meeting or calendar event.",
    requires_approval=True,
    tags=["calendar", "scheduling"],
)
def schedule_meeting(
    title: str,
    start: str,
    end: str,
    attendees: str,
    description: str = "",
) -> str:
    """Schedule a calendar meeting.

    Args:
        title: Meeting title / subject.
        start: Start time in ISO-8601 format (e.g. 2026-03-22T10:00:00).
        end: End time in ISO-8601 format.
        attendees: Comma-separated email addresses of attendees.
        description: Optional meeting description or agenda.

    Returns:
        Confirmation with event details.
    """
    if not title or not title.strip():
        return "Error: meeting title is required."
    if not start or not end:
        return "Error: start and end times are required."
    if not attendees or not attendees.strip():
        return "Error: at least one attendee is required."

    # Validate ISO format
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError as e:
        return f"Error: invalid datetime format -- {e}"

    if end_dt <= start_dt:
        return "Error: end time must be after start time."

    attendee_list = [a.strip() for a in attendees.split(",") if a.strip()]

    return (
        f"[PLACEHOLDER] Meeting scheduled successfully.\n"
        f"  Event ID: evt_{abs(hash(title + start)) % 100000:05d}\n"
        f"  Title: {title}\n"
        f"  Start: {start_dt.isoformat()}\n"
        f"  End: {end_dt.isoformat()}\n"
        f"  Duration: {end_dt - start_dt}\n"
        f"  Attendees: {', '.join(attendee_list)}\n"
        f"  Description: {description or '(none)'}\n"
        f"\n"
        f"Note: Configure CALENDAR_PROVIDER and credentials for real scheduling."
    )


@tool(
    name="check_availability",
    description="Check calendar availability for a given date and attendees.",
    tags=["calendar", "scheduling"],
)
def check_availability(date: str, attendees: str) -> str:
    """Check free/busy status for attendees on a given date.

    Args:
        date: Date to check in YYYY-MM-DD format.
        attendees: Comma-separated email addresses.

    Returns:
        JSON showing availability windows per attendee.
    """
    if not date or not date.strip():
        return "Error: date is required."
    if not attendees or not attendees.strip():
        return "Error: at least one attendee email is required."

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return "Error: date must be in YYYY-MM-DD format."

    attendee_list = [a.strip() for a in attendees.split(",") if a.strip()]

    availability = {}
    for attendee in attendee_list:
        availability[attendee] = {
            "date": date,
            "status": "free",
            "busy_slots": [
                {"start": f"{date}T09:00:00", "end": f"{date}T10:00:00", "title": "Team Standup"},
            ],
            "free_slots": [
                {"start": f"{date}T08:00:00", "end": f"{date}T09:00:00"},
                {"start": f"{date}T10:00:00", "end": f"{date}T17:00:00"},
            ],
        }

    return (
        f"[PLACEHOLDER] Availability for {date}.\n"
        f"Configure calendar provider for real free/busy data.\n\n"
        + json.dumps(availability, indent=2)
    )


@tool(
    name="list_events",
    description="List calendar events within a date range.",
    tags=["calendar", "scheduling"],
)
def list_events(start_date: str, end_date: str) -> str:
    """List events between two dates.

    Args:
        start_date: Start of range in YYYY-MM-DD format.
        end_date: End of range in YYYY-MM-DD format.

    Returns:
        JSON list of calendar events.
    """
    if not start_date or not end_date:
        return "Error: both start_date and end_date are required."

    try:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Error: dates must be in YYYY-MM-DD format."

    if e < s:
        return "Error: end_date must be on or after start_date."

    sample_events = [
        {
            "id": "evt_00001",
            "title": "Weekly Team Sync",
            "start": f"{start_date}T10:00:00",
            "end": f"{start_date}T10:30:00",
            "attendees": ["alice@example.com", "bob@example.com"],
            "recurring": True,
        },
        {
            "id": "evt_00002",
            "title": "Project Review",
            "start": f"{start_date}T14:00:00",
            "end": f"{start_date}T15:00:00",
            "attendees": ["alice@example.com"],
            "recurring": False,
        },
    ]

    return (
        f"[PLACEHOLDER] Events from {start_date} to {end_date}.\n"
        f"Configure calendar provider for real event data.\n\n"
        + json.dumps(sample_events, indent=2)
    )


@tool(
    name="cancel_event",
    description="Cancel a calendar event by event ID.",
    requires_approval=True,
    tags=["calendar", "scheduling"],
)
def cancel_event(event_id: str) -> str:
    """Cancel / delete a calendar event.

    Args:
        event_id: The unique identifier of the event to cancel.

    Returns:
        Confirmation or error message.
    """
    if not event_id or not event_id.strip():
        return "Error: event_id is required."

    return (
        f"[PLACEHOLDER] Event cancelled.\n"
        f"  Event ID: {event_id}\n"
        f"  Cancelled at: {datetime.now(timezone.utc).isoformat()}\n"
        f"\n"
        f"Note: Configure calendar provider for real cancellation."
    )


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_TOOLS: dict[str, Tool] = {
    "schedule_meeting": schedule_meeting,
    "check_availability": check_availability,
    "list_events": list_events,
    "cancel_event": cancel_event,
}


def get_calendar_tools(names: list[str] | None = None) -> list[Tool]:
    """Get calendar tools by name. If names is None, return all."""
    if names is None:
        return list(MODULE_TOOLS.values())
    return [MODULE_TOOLS[n] for n in names if n in MODULE_TOOLS]


try:
    from duxx_ai.tools.registry import register_domain

    register_domain("calendar", MODULE_TOOLS)
except ImportError:
    pass
