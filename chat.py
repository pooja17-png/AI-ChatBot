import streamlit as st
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define intents (patterns and responses)
intents = [
    {'intent': 'greeting', 'patterns': ['Hi', 'Hello', 'Hey', 'Good morning', 'Good evening'], 'responses': ['Hello! How can I help you?', 'Hi there! How can I assist you today?']},
    {'intent': 'goodbye', 'patterns': ['Bye', 'Goodbye', 'See you', 'Later'], 'responses': ['Goodbye! Take care!', 'See you later!']},
    {'intent': 'thankyou', 'patterns': ['Thank you', 'Thanks', 'Appreciate it'], 'responses': ['You’re welcome!', 'Glad I could help!']},
    {'intent': 'location', 'patterns': ['Where are you located?', 'What is your location?', 'Where are you?'], 'responses': ['I am in Pune, India.', 'I am based in Pune, India.']},
]

# Extract training data
patterns = []
responses = []
labels = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        labels.append(intent['intent'])

# Convert patterns into a bag-of-words format
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(patterns)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, labels)

# Function to get chatbot response based on user input
def chatbot_response(user_input):
    user_input_transformed = vectorizer.transform([user_input])
    intent_predicted = classifier.predict(user_input_transformed)[0]
    
    # Find the responses for the detected intent
    for intent in intents:
        if intent['intent'] == intent_predicted:
            # Choose a random response from the available responses for that intent
            return random.choice(intent['responses'])
    
    return "Sorry, I didn’t understand that."

# Streamlit UI
st.title("Simple Chatbot")

# User input
user_input = st.text_input("You:", "")

if user_input:
    # Get chatbot response
    bot_reply = chatbot_response(user_input)
    st.write(f"**Bot:** {bot_reply}")
