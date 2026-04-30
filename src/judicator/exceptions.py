class DetectionError(Exception):
    """Raised when judge type cannot be auto-detected from eval_template."""


class AuditCancelled(Exception):
    """Raised when user declines to proceed at the confirmation prompt."""


class FixtureNotFound(Exception):
    """Raised when a required fixture file does not exist."""
