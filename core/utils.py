"""Shared utility functions used across the codebase.

This module centralises common helpers to ensure consistent behaviour
and avoid duplication across modules.
"""

from __future__ import annotations

import re


# Pattern for slug validation: starts with lowercase letter, followed by lowercase letters, digits, or underscores
_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def is_slug(value: str) -> bool:
    """Check if a value matches the slug pattern [a-z][a-z0-9_]*.
    
    Used for validating step IDs, finding IDs, check IDs, and number keys.
    
    Args:
        value: The string to validate.
        
    Returns:
        True if the value matches the slug pattern, False otherwise.
    """
    if not isinstance(value, str):
        return False
    return bool(_SLUG_PATTERN.match(value))


def mask_dsn(dsn: str | None) -> str | None:
    """Mask sensitive parts of a database connection string.
    
    Keeps scheme, username, and host visible; hides only the password.
    The password (after : in user info) is replaced with *** for security.
    
    Args:
        dsn: The connection string to mask.
        
    Returns:
        A masked version of the DSN, or None if input is None.
    """
    if not dsn:
        return None
    scheme, separator, remainder = dsn.partition("://")
    if not separator:
        return "***"
    userinfo, at, host_path = remainder.partition("@")
    if not at:
        # No @ sign means no user info, but we should still mask the whole remainder
        return f"{scheme}://***"
    # Split userinfo into username and password
    username, colon, password = userinfo.partition(":")
    if colon and password:
        # Mask only the password
        masked_userinfo = f"{username}:***"
    else:
        # No password, just username
        masked_userinfo = username
    return f"{scheme}://{masked_userinfo}@{host_path}"


def truncate_text(text: str, budget: int) -> tuple[str, bool]:
    """Truncate text to fit within a budget with a safe truncation marker.
    
    Ensures the marker is never split and the result always fits within budget.
    Uses the same marker format as core.context.TRUNCATION_MARKER for consistency.
    
    Args:
        text: The text to truncate.
        budget: Maximum number of characters allowed.
        
    Returns:
        A tuple of (truncated_text, was_truncated).
    """
    if budget <= 0:
        return "", True
    if len(text) <= budget:
        return text, False
    
    # Use the same marker format as core.context.TRUNCATION_MARKER
    # The {omitted} placeholder is filled in below
    marker_template = "\n\n... [context truncated: {omitted} chars omitted] ...\n\n"
    
    # Iteratively compute the marker size since it depends on the omitted length
    marker = marker_template.format(omitted=len(text))
    for _ in range(4):
        head, tail = _split_budget(max(0, budget - len(marker)))
        candidate = marker_template.format(omitted=len(text) - head - tail)
        if len(candidate) == len(marker):
            marker = candidate
            break
        marker = candidate
    
    # Split available space between head (3/5) and tail (2/5)
    available = max(0, budget - len(marker))
    head, tail = _split_budget(available)
    
    head_text = text[:head]
    tail_text = text[-tail:] if tail > 0 else ""
    
    result = f"{head_text}{marker}{tail_text}"
    # Ensure we never exceed budget
    return result[:budget], True


def _split_budget(available: int) -> tuple[int, int]:
    """Split available space between head (3/5) and tail (2/5)."""
    head = (available * 3) // 5
    return head, available - head
