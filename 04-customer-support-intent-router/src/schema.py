"""Validated input/output contracts for the support router."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntentCategory(str, Enum):
    ACCOUNT_ACCESS = "account_access"
    BILLING_PAYMENT = "billing_payment"
    CANCELLATION = "cancellation"
    DELIVERY_STATUS = "delivery_status"
    REFUND_RETURN = "refund_return"
    SECURITY_PRIVACY = "security_privacy"
    TECHNICAL_ISSUE = "technical_issue"
    PRODUCT_QUESTION = "product_question"
    GENERAL_INQUIRY = "general_inquiry"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class SupportTicket(BaseModel):
    """Normalized ticket input. PII should be minimized before use."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1, max_length=100)
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=3, max_length=5000)
    channel: str = Field(default="web", min_length=1, max_length=40)
    customer_tier: str | None = Field(default=None, max_length=40)

    @field_validator("ticket_id", "subject", "message")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class IntentResult(BaseModel):
    """Schema-safe router response suitable for an API or queue."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    ticket_id: str = Field(min_length=1, max_length=100)
    category: IntentCategory
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_required: bool
    draft_response: str = Field(min_length=10, max_length=800)
    rationale: str = Field(min_length=3, max_length=240)
