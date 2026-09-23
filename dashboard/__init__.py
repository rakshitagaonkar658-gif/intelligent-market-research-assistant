"""
dashboard/__init__.py
──────────────────────
Dashboard package — exports all tab render functions.
"""

from dashboard.overview import render_overview
from dashboard.trends import render_trends
from dashboard.competitors import render_competitors
from dashboard.sentiment import render_sentiment
from dashboard.demand import render_demand
from dashboard.alerts import render_alerts
from dashboard.chat import render_chat

__all__ = [
    "render_overview",
    "render_trends",
    "render_competitors",
    "render_sentiment",
    "render_demand",
    "render_alerts",
    "render_chat",
]
