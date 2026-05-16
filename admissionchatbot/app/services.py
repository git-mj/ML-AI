from thefuzz import process, fuzz

class AdmissionChatbotService:
    faq_dataset = None

    @classmethod
    def load_model(cls):
        """Loads the FAQ dataset into memory."""
        if cls.faq_dataset is None:
            # Dictionary of 10 distinct admission questions/intents mapping to answers
            cls.faq_dataset = {
                "how to apply for admission online": "You can apply online through our university portal. The application fee is $50.",
                "what is the application deadline for fall spring": "The deadline for Fall admission is January 15th, and Spring is October 1st.",
                "what is the tuition cost and fees": "Tuition is approximately $15,000 per semester for in-state students.",
                "how to get scholarships and financial aid": "We offer merit-based scholarships. Please visit the Financial Aid page for the FAFSA requirements.",
                "is housing and dorms available on campus": "On-campus housing is guaranteed for all freshmen. Apply by May 1st.",
                "what majors and degree programs are offered": "We offer over 100 undergraduate majors including Computer Science, Business, and Engineering.",
                "how to book campus tours and visit": "Campus tours run daily at 10 AM and 2 PM. You can book online in advance.",
                "what are the international students requirements visa toefl": "International students must submit TOEFL scores and a student visa application.",
                "how to transfer credits from another college": "We accept transfer credits from accredited institutions. Please submit your official transcripts for evaluation.",
                "what are the admission requirements GPA SAT ACT": "We require a minimum 3.0 GPA. SAT/ACT scores are currently optional."
            }
            print("Admission Chatbot dataset loaded successfully.")

    @classmethod
    def get_answer(cls, text: str) -> dict:
        """
        Uses fuzzy matching to map a user's free-text question to the predefined dataset.
        Returns the answer or a fallback error message.
        """
        if not text or not text.strip():
            return {
                "answer": "Please ask a question.",
                "confidence": 0
            }

        text = text.lower()
        
        # Get list of possible intents (the questions/keywords in our dataset)
        intents = list(cls.faq_dataset.keys())
        
        # thefuzz process.extractOne returns (match_string, score)
        # Using token_set_ratio is good for ignoring extra words and matching subsets
        best_match = process.extractOne(text, intents, scorer=fuzz.token_set_ratio)
        
        if not best_match:
            confidence = 0
            matched_intent = None
        else:
            matched_intent = best_match[0]
            confidence = best_match[1]

        # If confidence is below 60%, we trigger the fallback response
        if confidence < 60:
            return {
                "answer": "I'm sorry, I don't have the answer to that question. Please reach out to our support team at admissions-support@university.edu for further assistance.",
                "confidence": confidence
            }
            
        # Return the corresponding answer
        return {
            "answer": cls.faq_dataset[matched_intent],
            "confidence": confidence
        }
