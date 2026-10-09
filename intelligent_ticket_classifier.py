"""
SWYNEX Internship - Task 3: Intelligent Feature
Project: Intelligent Ticket Classification with Error Handling, Confidence Thresholding & Fallback
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Dataset for Model Training & Validation
train_data = [
    # Billing
    ("I was charged twice for my subscription this month.", "Billing"),
    ("Need a refund for an unauthorized transaction on my credit card.", "Billing"),
    ("Where can I download my latest invoice and tax receipt?", "Billing"),
    ("My payment failed at checkout, but money was deducted.", "Billing"),
    
    # Technical Bug
    ("The mobile application crashes immediately when opening settings.", "Technical Bug"),
    ("Encountering an internal server error 500 when uploading files.", "Technical Bug"),
    ("The dashboard screen is blank and not displaying any data.", "Technical Bug"),
    ("The submit button is unresponsive and does nothing when clicked.", "Technical Bug"),
    
    # Account Access
    ("I forgot my password and the reset link email is not arriving.", "Account Access"),
    ("Unable to log in, receiving an invalid credentials error.", "Account Access"),
    ("How can I reset my two-factor authentication device on my profile?", "Account Access"),
    ("My account has been locked after several incorrect attempts.", "Account Access"),
    
    # Feature Request
    ("Could you please add dark mode support in the next desktop release?", "Feature Request"),
    ("It would be very helpful to have an option to export reports as PDF and CSV.", "Feature Request"),
    ("Please add support for multi-language localization in the app interface.", "Feature Request"),
    ("We need a bulk user import feature to upload contacts via CSV.", "Feature Request"),
]

# Separate features and labels
X_train = [text for text, label in train_data]
y_train = [label for text, label in train_data]

# 2. Build Pipeline
model = Pipeline([
    ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(random_state=42))
])
model.fit(X_train, y_train)

# 3. Intelligent Classification Function with Error Handling & Fallbacks
CONFIDENCE_THRESHOLD = 35.0  # Minimum percentage required for automated routing

def classify_ticket(ticket_text: str) -> dict:
    """
    Intelligent ticket classification with input validation,
    confidence thresholding, and automated human-fallback routing.
    """
    # Error Handling Case 1: Empty or Non-string Input
    if not ticket_text or not isinstance(ticket_text, str) or not ticket_text.strip():
        return {
            "status": "ERROR",
            "message": "Input validation failed: Ticket text cannot be empty.",
            "category": "Invalid Input",
            "confidence": 0.0,
            "routing": "Reject / Prompt User"
        }

    cleaned_text = ticket_text.strip()

    # Error Handling Case 2: Insufficient or Gibberish Input
    if len(cleaned_text.split()) < 2:
        return {
            "status": "ERROR",
            "message": "Input validation failed: Ticket must contain at least 2 descriptive words.",
            "category": "Insufficient Context",
            "confidence": 0.0,
            "routing": "Reject / Prompt User"
        }

    # Model Inference
    predicted_category = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    confidence = max(probabilities) * 100

    # Failure / Fallback Case: Low-Confidence Ambiguous Ticket
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "FALLBACK",
            "message": f"Low confidence ({confidence:.2f}% < {CONFIDENCE_THRESHOLD}%). Routed for manual review.",
            "category": predicted_category,
            "confidence": round(confidence, 2),
            "routing": "Human-in-the-Loop Manual Review"
        }

    # Successful High-Confidence Automated Routing
    return {
        "status": "SUCCESS",
        "message": "Automated triage completed successfully.",
        "category": predicted_category,
        "confidence": round(confidence, 2),
        "routing": f"Automated Routing to [{predicted_category} Team]"
    }


# 4. Evaluation and Failure Cases Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("   SWYNEX TASK 3: INTELLIGENT TICKET CLASSIFIER EVALUATION & DEMO")
    print("=" * 70)

    test_suite = [
        # Standard Valid Inputs
        "I was billed twice on my monthly renewal.",
        "The app freezes and crashes whenever I click export.",
        "Can you add support for webhook notifications in future versions?",
        
        # Failure & Edge Cases (Error Handling)
        "",                                          # Empty string
        "Help",                                      # Too short / vague
        "Hi there, I am wondering if you might be having nice weather today?" # Ambiguous / Out-of-domain
    ]

    for item in test_suite:
        result = classify_ticket(item)
        print(f"\n[Ticket Input]     : \"{item}\"")
        print(f" -> Status         : {result['status']}")
        print(f" -> Category       : {result['category']}")
        print(f" -> Confidence     : {result['confidence']}%")
        print(f" -> Routing Action : {result['routing']}")
        print(f" -> Diagnostic     : {result['message']}")

    print("\n" + "=" * 70)
