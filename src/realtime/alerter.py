"""
Alert Manager Module
=====================

This module provides alerting capabilities for deepfake detection events.

Features:
    - Multiple notification channels (webhook, email, file)
    - Alert severity levels
    - Rate limiting to prevent alert storms
    - Alert history and aggregation
    - Customizable alert templates

Example Usage:
    >>> alerter = AlertManager(webhook_url="https://slack.com/webhook")
    >>> alerter.send_alert(detection_event)
"""

from __future__ import annotations

import json
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from collections import deque

from src.utils.logging import get_logger
from src.utils.exceptions import ServiceError

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        priorities = {
            AlertLevel.INFO: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.CRITICAL: 3,
            AlertLevel.EMERGENCY: 4,
        }
        return priorities[self]


@dataclass
class Alert:
    """
    Container for alert information.

    Attributes:
        alert_id: Unique alert identifier.
        timestamp: Alert creation time.
        level: Alert severity level.
        source: Alert source identifier.
        title: Alert title.
        message: Detailed message.
        details: Additional details.
        acknowledged: Whether alert was acknowledged.
    """
    alert_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    level: AlertLevel = AlertLevel.WARNING
    source: str = ""
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"alert_{int(self.timestamp.timestamp() * 1000)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "source": self.source,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AlertConfig:
    """
    Configuration for alert manager.

    Attributes:
        rate_limit_window: Time window for rate limiting (seconds).
        rate_limit_max: Maximum alerts per window.
        min_interval: Minimum interval between similar alerts (seconds).
        aggregation_window: Window for alert aggregation (seconds).
        retention_hours: Hours to retain alert history.
    """
    rate_limit_window: int = 60
    rate_limit_max: int = 10
    min_interval: int = 30
    aggregation_window: int = 300
    retention_hours: int = 24


# =============================================================================
# Alert Channels
# =============================================================================

class AlertChannel:
    """Base class for alert channels."""

    def send(self, alert: Alert) -> bool:
        """
        Send an alert through this channel.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        raise NotImplementedError


class WebhookChannel(AlertChannel):
    """
    Webhook alert channel (Slack, Discord, etc.).

    Sends alerts via HTTP POST to a webhook URL.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        template: Optional[Callable[[Alert], Dict]] = None,
    ) -> None:
        """
        Initialize webhook channel.

        Args:
            url: Webhook URL.
            headers: Custom HTTP headers.
            template: Function to format alert payload.
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.template = template or self._default_template

    def _default_template(self, alert: Alert) -> Dict[str, Any]:
        """Default Slack-compatible template."""
        color_map = {
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ff9800",
            AlertLevel.CRITICAL: "#f44336",
            AlertLevel.EMERGENCY: "#9c27b0",
        }

        return {
            "attachments": [{
                "color": color_map.get(alert.level, "#808080"),
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Level", "value": alert.level.value, "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                ],
                "footer": f"Alert ID: {alert.alert_id}",
            }]
        }

    def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import urllib.request

            payload = self.template(alert)
            data = json.dumps(payload).encode("utf-8")

            request = urllib.request.Request(
                self.url,
                data=data,
                headers=self.headers,
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status == 200

        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False


class EmailChannel(AlertChannel):
    """
    Email alert channel.

    Sends alerts via SMTP email.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True,
    ) -> None:
        """
        Initialize email channel.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port.
            username: SMTP username.
            password: SMTP password.
            from_addr: Sender email address.
            to_addrs: Recipient email addresses.
            use_tls: Use TLS encryption.
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            # Plain text body
            text_body = f"""
Deepfake Detection Alert
========================

Level: {alert.level.value.upper()}
Source: {alert.source}
Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

{alert.message}

Details:
{json.dumps(alert.details, indent=2)}

Alert ID: {alert.alert_id}
            """.strip()

            # HTML body
            html_body = f"""
<html>
<body>
<h2>Deepfake Detection Alert</h2>
<table>
    <tr><td><strong>Level:</strong></td><td>{alert.level.value.upper()}</td></tr>
    <tr><td><strong>Source:</strong></td><td>{alert.source}</td></tr>
    <tr><td><strong>Time:</strong></td><td>{alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
</table>
<p>{alert.message}</p>
<pre>{json.dumps(alert.details, indent=2)}</pre>
<hr>
<small>Alert ID: {alert.alert_id}</small>
</body>
</html>
            """.strip()

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            return True

        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class FileChannel(AlertChannel):
    """
    File-based alert channel.

    Writes alerts to a file (JSON lines format).
    """

    def __init__(
        self,
        file_path: str,
        max_size_mb: int = 100,
        rotate: bool = True,
    ) -> None:
        """
        Initialize file channel.

        Args:
            file_path: Path to alert file.
            max_size_mb: Maximum file size before rotation.
            rotate: Enable file rotation.
        """
        self.file_path = Path(file_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.rotate = rotate
        self._lock = threading.Lock()

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> bool:
        """Write alert to file."""
        try:
            with self._lock:
                # Check for rotation
                if self.rotate and self.file_path.exists():
                    if self.file_path.stat().st_size > self.max_size_bytes:
                        self._rotate_file()

                # Append alert
                with open(self.file_path, "a") as f:
                    f.write(alert.to_json().replace("\n", " ") + "\n")

            return True

        except Exception as e:
            logger.error(f"File write failed: {e}")
            return False

    def _rotate_file(self) -> None:
        """Rotate the alert file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = self.file_path.with_suffix(f".{timestamp}.json")
        self.file_path.rename(rotated_path)
        logger.info(f"Rotated alert file to: {rotated_path}")


class CallbackChannel(AlertChannel):
    """
    Callback-based alert channel.

    Invokes a custom callback function for each alert.
    """

    def __init__(self, callback: Callable[[Alert], None]) -> None:
        """
        Initialize callback channel.

        Args:
            callback: Function to call with each alert.
        """
        self.callback = callback

    def send(self, alert: Alert) -> bool:
        """Send alert via callback."""
        try:
            self.callback(alert)
            return True
        except Exception as e:
            logger.error(f"Callback failed: {e}")
            return False


# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
    """
    Manager for handling detection alerts.

    This class manages alert channels, rate limiting,
    aggregation, and alert history.

    Attributes:
        config: Alert configuration.
        channels: Registered alert channels.

    Example:
        >>> alerter = AlertManager(webhook_url="https://slack.com/webhook")
        >>> alerter.add_channel(EmailChannel(...))
        >>> alerter.send_alert(detection_event)
    """

    def __init__(
        self,
        config: Optional[AlertConfig] = None,
        webhook_url: Optional[str] = None,
        alert_file: Optional[str] = None,
    ) -> None:
        """
        Initialize alert manager.

        Args:
            config: Alert configuration.
            webhook_url: Optional webhook URL for quick setup.
            alert_file: Optional file path for alert logging.
        """
        self.config = config or AlertConfig()
        self.channels: List[AlertChannel] = []
        self._alert_history: deque = deque(maxlen=1000)
        self._rate_limit_window: deque = deque()
        self._last_alert_by_source: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        # Add channels from constructor arguments
        if webhook_url:
            self.add_channel(WebhookChannel(webhook_url))

        if alert_file:
            self.add_channel(FileChannel(alert_file))

        logger.info("AlertManager initialized")

    def add_channel(self, channel: AlertChannel) -> None:
        """
        Add an alert channel.

        Args:
            channel: Alert channel to add.
        """
        self.channels.append(channel)
        logger.info(f"Added alert channel: {type(channel).__name__}")

    def remove_channel(self, channel: AlertChannel) -> None:
        """
        Remove an alert channel.

        Args:
            channel: Alert channel to remove.
        """
        if channel in self.channels:
            self.channels.remove(channel)

    def send_alert(
        self,
        event: Any,
        level: Optional[AlertLevel] = None,
    ) -> Optional[Alert]:
        """
        Send an alert for a detection event.

        Args:
            event: Detection event or Alert object.
            level: Override alert level.

        Returns:
            Alert if sent, None if rate limited.
        """
        with self._lock:
            # Convert event to alert if needed
            if isinstance(event, Alert):
                alert = event
            else:
                alert = self._create_alert_from_event(event, level)

            # Check rate limiting
            if not self._check_rate_limit(alert):
                logger.debug(f"Alert rate limited: {alert.source}")
                return None

            # Send to all channels
            success_count = 0
            for channel in self.channels:
                try:
                    if channel.send(alert):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Channel send error: {e}")

            # Record alert
            self._alert_history.append(alert)
            self._last_alert_by_source[alert.source] = alert.timestamp

            logger.info(
                f"Alert sent to {success_count}/{len(self.channels)} channels: "
                f"{alert.title}"
            )

            return alert

    def _create_alert_from_event(
        self,
        event: Any,
        level: Optional[AlertLevel] = None,
    ) -> Alert:
        """Create alert from detection event."""
        # Determine alert level from probability
        if level is None:
            prob = getattr(event, "probability", 0.5)
            if prob >= 0.9:
                level = AlertLevel.EMERGENCY
            elif prob >= 0.8:
                level = AlertLevel.CRITICAL
            elif prob >= 0.6:
                level = AlertLevel.WARNING
            else:
                level = AlertLevel.INFO

        return Alert(
            level=level,
            source=getattr(event, "stream_url", "unknown"),
            title=f"Deepfake Detected - {level.value.upper()}",
            message=(
                f"Potential deepfake detected with {getattr(event, 'probability', 0):.1%} "
                f"probability (confidence: {getattr(event, 'confidence', 0):.1%})"
            ),
            details=event.to_dict() if hasattr(event, "to_dict") else {},
        )

    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        now = datetime.now()

        # Clean old entries from rate limit window
        cutoff = now - timedelta(seconds=self.config.rate_limit_window)
        while self._rate_limit_window and self._rate_limit_window[0] < cutoff:
            self._rate_limit_window.popleft()

        # Check rate limit
        if len(self._rate_limit_window) >= self.config.rate_limit_max:
            return False

        # Check minimum interval for same source
        if alert.source in self._last_alert_by_source:
            last_time = self._last_alert_by_source[alert.source]
            if (now - last_time).total_seconds() < self.config.min_interval:
                return False

        # Record this alert in rate limit window
        self._rate_limit_window.append(now)
        return True

    def get_alert_history(
        self,
        limit: int = 100,
        level: Optional[AlertLevel] = None,
        source: Optional[str] = None,
    ) -> List[Alert]:
        """
        Get recent alert history.

        Args:
            limit: Maximum alerts to return.
            level: Filter by level.
            source: Filter by source.

        Returns:
            List of recent alerts.
        """
        alerts = list(self._alert_history)

        if level:
            alerts = [a for a in alerts if a.level == level]

        if source:
            alerts = [a for a in alerts if a.source == source]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge.

        Returns:
            True if alert was found and acknowledged.
        """
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        recent_alerts = [
            a for a in self._alert_history
            if a.timestamp > hour_ago
        ]

        return {
            "total_alerts": len(self._alert_history),
            "alerts_last_hour": len(recent_alerts),
            "by_level": {
                level.value: len([a for a in recent_alerts if a.level == level])
                for level in AlertLevel
            },
            "channels": len(self.channels),
        }


# =============================================================================
# Service Factory
# =============================================================================

def create_alert_manager(
    config: Optional[Dict[str, Any]] = None,
) -> AlertManager:
    """
    Factory function to create an AlertManager.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured AlertManager instance.
    """
    if config is None:
        config = {}

    alert_config = AlertConfig(
        rate_limit_window=config.get("rate_limit_window", 60),
        rate_limit_max=config.get("rate_limit_max", 10),
        min_interval=config.get("min_interval", 30),
    )

    manager = AlertManager(
        config=alert_config,
        webhook_url=config.get("webhook_url"),
        alert_file=config.get("alert_file"),
    )

    # Add email channel if configured
    if "email" in config:
        email_config = config["email"]
        manager.add_channel(EmailChannel(
            smtp_host=email_config["smtp_host"],
            smtp_port=email_config["smtp_port"],
            username=email_config["username"],
            password=email_config["password"],
            from_addr=email_config["from_addr"],
            to_addrs=email_config["to_addrs"],
        ))

    return manager
