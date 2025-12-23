def __init__(self):
    self.context = {}

def respond(self, message: str) -> str:
    """
    Provides a response to a user's message.

    Args:
        message (str): The user's message.

    Returns:
        str: The chatbot's response.
    """
    message = message.lower()

    if "hello" in message or "hi" in message:
        return "Hello! How can I assist you today?"

    if "how are you" in message:
        return "I'm just a computer program, but I'm functioning well. Thanks for asking!"

    if "what is your name" in message:
        return "I'm the Kaleidoscope AI System Chatbot. You can call me Kaleidoscope."

    if "tell me a joke" in message:
        return "Why don't scientists trust atoms? Because they make up everything!"

    if "thank you" in message or "thanks" in message:
        return "You're welcome! Is there anything else I can help with?"

    if "bye" in message or "goodbye" in message:
        return "Goodbye! Have a great day!"

    # Example of context-aware response
    if "what did i say" in message:
        if "last_message" in self.context:
            return f"You said: '{self.context['last_message']}'"
        else:
            return "I don't have any previous messages from you in my context."

    # Update context with the last message
    self.context["last_message"] = message

    return "I'm not sure how to respond to that. Can you try a different question?"


