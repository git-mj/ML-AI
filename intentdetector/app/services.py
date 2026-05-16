from thefuzz import process, fuzz

class IntentDetectorService:
    dataset = None

    @classmethod
    def load_model(cls):
        """Loads the domain and intent dataset into memory."""
        if cls.dataset is None:
            # Structure: Domain -> { Intent: [Keywords] }
            cls.dataset = {
                "Hardware": {
                    "Replace": ["broken", "shattered", "cracked", "replacement", "new laptop", "monitor dead", "replace"],
                    "Repair": ["fix", "repair", "not turning on", "battery dead", "overheating", "loud fan"],
                    "Peripherals": ["mouse", "keyboard", "headset", "docking station", "printer", "printing"]
                },
                "Software": {
                    "Install": ["install", "download", "setup", "need access", "get app", "software request"],
                    "Update": ["update", "upgrade", "new version", "patch", "outdated"],
                    "License": ["license key", "activation", "expired", "renew", "product key"]
                },
                "Access": {
                    "Password Reset": ["password", "locked out", "reset", "forgot password", "unlock", "credentials"],
                    "VPN": ["vpn", "remote", "network", "cisco", "anyconnect", "offline", "connection"],
                    "Permissions": ["access denied", "permissions", "admin rights", "shared drive", "folder access"]
                }
            }
            print("Intent Detector dataset loaded successfully.")

    @classmethod
    def detect_intent(cls, text: str) -> dict:
        """
        Detects the domain and intent using fuzzy matching to account for typos.
        """
        if not text or not text.strip():
            return {"domain": "Unknown", "intent": "Unknown", "confidence": 0}

        text = text.lower()
        
        domain_scores = {}
        for domain, intents in cls.dataset.items():
            domain_keywords = []
            for keywords in intents.values():
                domain_keywords.extend(keywords)
            
            # Find the best matching keyword in this domain for the given text
            # thefuzz process.extractOne returns (match_string, score)
            best_match = process.extractOne(text, domain_keywords, scorer=fuzz.token_set_ratio)
            if best_match:
                domain_scores[domain] = best_match[1]
            else:
                domain_scores[domain] = 0

        # Determine the best domain
        best_domain = max(domain_scores, key=domain_scores.get)
        domain_confidence = domain_scores[best_domain]

        # If confidence is too low, we classify it as unknown
        if domain_confidence < 40:
            return {"domain": "Unknown", "intent": "Unknown", "confidence": domain_confidence}

        # Identify intent within the best domain
        intent_scores = {}
        domain_intents = cls.dataset[best_domain]
        for intent, keywords in domain_intents.items():
            best_intent_match = process.extractOne(text, keywords, scorer=fuzz.token_set_ratio)
            if best_intent_match:
                intent_scores[intent] = best_intent_match[1]
            else:
                intent_scores[intent] = 0

        best_intent = max(intent_scores, key=intent_scores.get)
        intent_confidence = intent_scores[best_intent]

        return {
            "domain": best_domain,
            "intent": best_intent,
            "confidence": intent_confidence
        }
