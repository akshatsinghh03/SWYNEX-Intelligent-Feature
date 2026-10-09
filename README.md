# SWYNEX - Intelligent Feature (Task 3)

## Project Overview
This repository enhances the customer support ticket classification system built in Task 2 by adding **intelligent production-grade features**, including input validation, confidence-based fallback mechanisms (Human-in-the-Loop), failure case detection, and structured evaluation.

---

## Intelligent Features Added
1. **Input Validation & Error Handling:** Rejects empty or one-word messages that lack context.
2. **Confidence Thresholding (<35% fallback):** Prevents misleading classifications on ambiguous or out-of-domain queries by routing them to manual human review.
3. **Structured Diagnostics:** Outputs status codes (`SUCCESS`, `FALLBACK`, `ERROR`) alongside confidence scores and target routing queues.

---

## Evaluation & Test Cases

| Case Type | Input Text | Status | Routing Action | Diagnostic Note |
|-----------|------------|--------|----------------|-----------------|
| **Normal Valid** | *"I was billed twice on my monthly renewal."* | `SUCCESS` | Automated Routing to [Billing Team] | High confidence match |
| **Normal Valid** | *"The app freezes and crashes whenever I click export."* | `SUCCESS` | Automated Routing to [Technical Bug Team] | High confidence match |
| **Edge Case (Empty)** | `""` | `ERROR` | Reject / Prompt User | Input cannot be empty |
| **Edge Case (Vague)** | `"Help"` | `ERROR` | Reject / Prompt User | Requires $\ge 2$ descriptive words |
| **Failure / Fallback** | *"Hi there, I am wondering if you might be having nice weather today?"* | `FALLBACK` | Human-in-the-Loop Manual Review | Low confidence out-of-domain query |

---

## How to Run
1. Install dependencies:
   ```bash
   pip install scikit-learn
