# Synthetic Shipment Status Guide

This guide is fictional training material for demonstrating shipment lookup and retrieval. The operational status stored in a shipment record is more authoritative for that record than this generic explanation.

## In transit

“In transit” means a carrier scan indicates the shipment is moving between operational points. The location and update timestamp in the record describe the latest synthetic event. Do not infer an exact vehicle location or guarantee an arrival time from this status alone.

## Delivered

“Delivered” means the synthetic record contains a destination delivery scan. If a recipient disputes delivery, preserve the tracking reference and escalate to the operations team for a delivery proof review. This example knowledge base does not contain recipient signatures or photographs.

## Awaiting pickup

“Awaiting pickup” means a pickup request exists but a collection scan has not yet been recorded. Verify the pickup window and collection instructions. If the expected pickup window has passed, contact the assigned operations queue rather than fabricating a collection event.

## Exception

“Exception” indicates that the latest synthetic update requires manual follow-up, such as address validation or a facility hold. Check the exception note and route it to the responsible operations team. Do not treat an exception as proof that a shipment is lost.
