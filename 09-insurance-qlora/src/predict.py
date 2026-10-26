"""Rules-baseline claim routing CLI; use only synthetic inputs in this demo."""
import argparse
import json
from .baseline import route_claim
from .schema import CLAIM_TYPES


def main() -> int:
    parser = argparse.ArgumentParser(description="Route a synthetic claim for human review; no claim outcome is decided.")
    parser.add_argument("--claim-type", choices=sorted(CLAIM_TYPES), required=True)
    parser.add_argument("--description", default="Synthetic claim record for demonstration")
    parser.add_argument("--claim-id", default="SYN-USER01")
    parser.add_argument("--documents", default="", help="Comma-separated submitted document names")
    args = parser.parse_args()
    documents = [item.strip() for item in args.documents.split(",") if item.strip()]
    result = route_claim(args.claim_type, args.claim_id, documents, args.description)
    result["input_description"] = args.description
    result["disclaimer"] = "Synthetic demonstration only. Human review is required; no coverage, liability, fraud, payout, or denial decision is made."
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
