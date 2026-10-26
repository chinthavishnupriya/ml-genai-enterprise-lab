"""Command-line entry point for the Customer Support Intent Router."""
import argparse
import json
import sys
from .schema import SupportTicket
from .router import SupportIntentRouter
from .llm_provider import OpenAICompatibleRouter, ProviderError


def main() -> int:
    parser = argparse.ArgumentParser(description="Route a support ticket to an intent and priority.")
    parser.add_argument("--ticket-id", default="T-1001")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--channel", default="web")
    parser.add_argument("--provider", choices=("rules", "openai"), default="rules")
    args = parser.parse_args()
    ticket = SupportTicket(ticket_id=args.ticket_id, subject=args.subject, message=args.message, channel=args.channel)
    router = OpenAICompatibleRouter() if args.provider == "openai" else SupportIntentRouter()
    try:
        result = router.route(ticket)
    except ProviderError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
