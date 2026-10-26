"""Deterministic offline benchmark cases for intent and priority routing."""
from dataclasses import dataclass
from .schema import IntentCategory, Priority, SupportTicket
from .router import SupportIntentRouter


@dataclass(frozen=True)
class BenchmarkCase:
    ticket: SupportTicket
    expected_category: IntentCategory
    expected_priority: Priority


CASES = [
    BenchmarkCase(SupportTicket(ticket_id="B001", subject="Account hacked", message="Someone accessed my account and changed the password."), IntentCategory.SECURITY_PRIVACY, Priority.URGENT),
    BenchmarkCase(SupportTicket(ticket_id="B002", subject="Suspicious login", message="I see a suspicious login I did not authorize."), IntentCategory.SECURITY_PRIVACY, Priority.URGENT),
    BenchmarkCase(SupportTicket(ticket_id="B003", subject="Charged twice", message="My card was charged twice for the same order."), IntentCategory.BILLING_PAYMENT, Priority.HIGH),
    BenchmarkCase(SupportTicket(ticket_id="B004", subject="Need invoice", message="Please send the invoice and receipt for my payment."), IntentCategory.BILLING_PAYMENT, Priority.MEDIUM),
    BenchmarkCase(SupportTicket(ticket_id="B005", subject="Password reset", message="I forgot my password and cannot log in."), IntentCategory.ACCOUNT_ACCESS, Priority.MEDIUM),
    BenchmarkCase(SupportTicket(ticket_id="B006", subject="Locked out", message="I am locked out of my account and need access."), IntentCategory.ACCOUNT_ACCESS, Priority.HIGH),
    BenchmarkCase(SupportTicket(ticket_id="B007", subject="Cancel plan", message="Please cancel my subscription before it renews."), IntentCategory.CANCELLATION, Priority.LOW),
    BenchmarkCase(SupportTicket(ticket_id="B008", subject="Stop renewal", message="Turn off auto-renew for my membership."), IntentCategory.CANCELLATION, Priority.LOW),
    BenchmarkCase(SupportTicket(ticket_id="B009", subject="Track order", message="Where is my order? The tracking number has not updated."), IntentCategory.DELIVERY_STATUS, Priority.MEDIUM),
    BenchmarkCase(SupportTicket(ticket_id="B010", subject="Missing package", message="My package is missing and the delivery was marked complete."), IntentCategory.DELIVERY_STATUS, Priority.HIGH),
    BenchmarkCase(SupportTicket(ticket_id="B011", subject="Refund request", message="I would like a refund and money back for this purchase."), IntentCategory.REFUND_RETURN, Priority.MEDIUM),
    BenchmarkCase(SupportTicket(ticket_id="B012", subject="Return label", message="Please provide a return label for the wrong item."), IntentCategory.REFUND_RETURN, Priority.MEDIUM),
    BenchmarkCase(SupportTicket(ticket_id="B013", subject="Application crash", message="The app keeps crashing every time I upload a file."), IntentCategory.TECHNICAL_ISSUE, Priority.HIGH),
    BenchmarkCase(SupportTicket(ticket_id="B014", subject="Error message", message="I get a 500 error message when the page loads."), IntentCategory.TECHNICAL_ISSUE, Priority.MEDIUM),
    BenchmarkCase(SupportTicket(ticket_id="B015", subject="Compatibility", message="Is this device compatible with supported formats listed in the manual?"), IntentCategory.PRODUCT_QUESTION, Priority.LOW),
    BenchmarkCase(SupportTicket(ticket_id="B016", subject="Warranty question", message="What is the warranty period for this product?"), IntentCategory.PRODUCT_QUESTION, Priority.LOW),
    BenchmarkCase(SupportTicket(ticket_id="B017", subject="Support hours", message="Can you tell me your business hours?"), IntentCategory.GENERAL_INQUIRY, Priority.LOW),
    BenchmarkCase(SupportTicket(ticket_id="B018", subject="Contact an agent", message="How can I contact support to ask a general question?"), IntentCategory.GENERAL_INQUIRY, Priority.LOW),
]


def run_benchmark(router: SupportIntentRouter | None = None) -> dict:
    router = router or SupportIntentRouter()
    category_correct = priority_correct = exact_correct = 0
    misclassifications = []
    for case in CASES:
        result = router.route(case.ticket)
        cat_ok = result.category == case.expected_category
        prio_ok = result.priority == case.expected_priority
        category_correct += int(cat_ok)
        priority_correct += int(prio_ok)
        exact_correct += int(cat_ok and prio_ok)
        if not (cat_ok and prio_ok):
            misclassifications.append({
                "ticket_id": case.ticket.ticket_id,
                "expected_category": case.expected_category.value,
                "predicted_category": result.category,
                "expected_priority": case.expected_priority.value,
                "predicted_priority": result.priority,
            })
    count = len(CASES)
    return {"cases": count, "category_accuracy": round(category_correct / count, 4),
            "priority_accuracy": round(priority_correct / count, 4),
            "exact_match_accuracy": round(exact_correct / count, 4),
            "misclassifications": misclassifications,
            "note": "Curated synthetic rule-coverage benchmark; not an estimate of real-world customer-support accuracy."}


def main() -> int:
    import json
    print(json.dumps(run_benchmark(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
