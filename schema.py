# schema.py
from pydantic import BaseModel, Field, conlist
from typing import Literal, Optional, List, Union

class Trigger(BaseModel):
    type: Literal["event", "schedule", "conditional"] = Field(..., description="The type of trigger.")
    source: Optional[str] = Field(None, description="The source of the event, e.g., 'monitoring_agent.server_metrics'.")
    expression: Optional[str] = Field(None, description="The conditional expression, e.g., 'cpu.load > 0.8'.")
    duration_seconds: Optional[int] = Field(None, description="Duration the condition must hold true for, in seconds.")
    cron_expression: Optional[str] = Field(None, description="A cron expression for scheduled triggers, e.g., '0 * * * *'.")

class Action(BaseModel):
    type: Literal["api_call", "notification", "script"] = Field(..., description="The type of action to perform.")
    target: str = Field(..., description="The target of the action, e.g., an API endpoint or a script name.")
    payload: Optional[dict] = Field({}, description="The data to send with the action, e.g., {'service': 'web-workers', 'replicas': 5}.")
    recipient: Optional[str] = Field(None, description="The recipient for notifications, e.g., 'on-call-slack-channel'.")

class StructuredCommand(BaseModel):
    """A machine-executable command translated from a natural language request."""
    description: str = Field(..., description="A summary of the command's purpose.")
    trigger: Trigger = Field(..., description="The event or condition that initiates the command.")
    actions: conlist(Action, min_length=1) = Field(..., description="A list of one or more actions to be executed when triggered.")
