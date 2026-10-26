"""Deterministic, offline-first intent and priority routing."""
import re
from typing import Iterable
from .schema import IntentCategory, IntentResult, Priority, SupportTicket

CATEGORY_PATTERNS: dict[IntentCategory, tuple[str, ...]] = {
    IntentCategory.SECURITY_PRIVACY: ("hacked", "unauthorized", "account was compromised", "data breach", "stolen card", "suspicious login", "someone accessed", "security incident", "identity theft", "exposed my data", "phishing", "fraudulent activity"),
    IntentCategory.REFUND_RETURN: ("refund", "money back", "return this", "return label", "reimbursement", "send it back", "wrong item", "damaged item", "replacement or refund"),
    IntentCategory.BILLING_PAYMENT: ("billing", "invoice", "charged", "charge", "payment", "receipt", "subscription fee", "credit card statement", "tax on my bill", "double billed"),
    IntentCategory.ACCOUNT_ACCESS: ("can't log in", "cannot log in", "can't login", "cannot login", "sign in", "password reset", "reset my password", "locked out", "account locked", "forgot my password", "login code", "two-factor", "2fa code"),
    IntentCategory.CANCELLATION: ("cancel", "cancellation", "terminate my plan", "close my subscription", "stop renewal", "turn off auto-renew", "end my membership"),
    IntentCategory.DELIVERY_STATUS: ("where is my order", "track my order", "tracking number", "shipment", "delivery status", "hasn't arrived", "not arrived", "package is late", "parcel", "shipping update", "lost package", "package is missing"),
    IntentCategory.TECHNICAL_ISSUE: ("app crashes", "keeps crashing", "error message", "bug", "not loading", "page won't load", "cannot upload", "can't upload", "service is down", "server error", "technical issue", "not working", "500 error", "sync failed"),
    IntentCategory.PRODUCT_QUESTION: ("does this support", "is it compatible", "compatibility", "product features", "how does this work", "specifications", "available in", "what is included", "supported formats", "warranty period", "size guide"),
    IntentCategory.GENERAL_INQUIRY: ("business hours", "contact support", "speak to an agent", "help me", "general question", "information about", "how can i contact", "thank you"),
}
URGENT_PATTERNS = ("hacked", "account was compromised", "unauthorized access", "data breach", "stolen card", "identity theft", "fraudulent activity", "someone accessed my account", "security incident", "active fraud", "suspicious login", "did not authorize")
HIGH_PATTERNS = ("charged twice", "double billed", "locked out", "cannot access my account", "can't access my account", "service is down", "production outage", "all users", "keeps crashing", "deadline today", "payment failed at checkout", "lost package", "package is missing", "can't access payroll", "cannot access payroll")

DRAFTS: dict[IntentCategory, str] = {
    IntentCategory.ACCOUNT_ACCESS: "Thanks for contacting support. We can help with account access. Please use the official password-reset or sign-in recovery flow, and do not share your password or one-time codes in this ticket.",
    IntentCategory.BILLING_PAYMENT: "Thanks for flagging this billing question. We will review the charge and related invoice details. Please do not send full payment-card numbers; the transaction date and last four digits are usually sufficient for a secure follow-up.",
    IntentCategory.CANCELLATION: "Thanks for contacting us about cancellation. We will check the current subscription status and applicable end date. No cancellation is considered complete until it is confirmed in your account.",
    IntentCategory.DELIVERY_STATUS: "Thanks for checking on your delivery. We will review the latest tracking information and available carrier updates. Please include the order reference, but avoid posting personal address details in a public channel.",
    IntentCategory.REFUND_RETURN: "Thanks for contacting us about a return or refund. We will check the order and the applicable return policy, then confirm the available next steps. This message is not a confirmation that a refund has been issued.",
    IntentCategory.SECURITY_PRIVACY: "We take this security concern seriously. Please use the official account-security flow to secure your account and contact the verified support channel. Do not share passwords, one-time codes, or full payment details here.",
    IntentCategory.TECHNICAL_ISSUE: "Thanks for reporting the technical issue. We will investigate the behavior. If possible, share the error text, approximate time, affected feature, and device or browser version; remove passwords, tokens, and other secrets first.",
    IntentCategory.PRODUCT_QUESTION: "Thanks for your question. We will confirm the relevant product specifications and compatibility details so that we can give you an accurate answer.",
    IntentCategory.GENERAL_INQUIRY: "Thanks for reaching out. We will review your question and direct it to the appropriate support team if needed. Please avoid including sensitive personal information unless the verified support process requires it.",
}


def _matching_phrases(text: str, phrases: Iterable[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def classify_category(ticket: SupportTicket) -> tuple[IntentCategory, float, list[str]]:
    text = f"{ticket.subject} {ticket.message}".casefold()
    hits = {category: _matching_phrases(text, phrases) for category, phrases in CATEGORY_PATTERNS.items()}
    if hits[IntentCategory.SECURITY_PRIVACY]:
        return IntentCategory.SECURITY_PRIVACY, 0.97, hits[IntentCategory.SECURITY_PRIVACY]
    ranked = sorted(hits.items(), key=lambda item: (len(item[1]), item[0].value), reverse=True)
    category, phrases = ranked[0]
    if not phrases:
        return IntentCategory.GENERAL_INQUIRY, 0.52, []
    confidence = min(0.94, 0.68 + 0.08 * (len(phrases) - 1))
    return category, round(confidence, 2), phrases


def classify_priority(ticket: SupportTicket, category: IntentCategory) -> Priority:
    text = f"{ticket.subject} {ticket.message}".casefold()
    if _matching_phrases(text, URGENT_PATTERNS):
        return Priority.URGENT
    if _matching_phrases(text, HIGH_PATTERNS):
        return Priority.HIGH
    if category in {IntentCategory.PRODUCT_QUESTION, IntentCategory.GENERAL_INQUIRY, IntentCategory.CANCELLATION}:
        return Priority.LOW
    if re.search(r"\b(asap|immediately|urgent|today's deadline|critical)\b", text):
        return Priority.HIGH
    return Priority.MEDIUM


class SupportIntentRouter:
    """Offline-first rule router; deterministic and credential-free."""

    def route(self, ticket: SupportTicket) -> IntentResult:
        category, confidence, matched = classify_category(ticket)
        priority = classify_priority(ticket, category)
        escalation = priority in {Priority.HIGH, Priority.URGENT} or category == IntentCategory.SECURITY_PRIVACY
        matched_text = ", ".join(matched[:3]) if matched else "no category keyword matched; safe general queue used"
        rationale = f"Rule match: {matched_text}. Priority set to {priority.value} by severity rules."
        return IntentResult(ticket_id=ticket.ticket_id, category=category, priority=priority, confidence=confidence, escalation_required=escalation, draft_response=DRAFTS[category], rationale=rationale)
