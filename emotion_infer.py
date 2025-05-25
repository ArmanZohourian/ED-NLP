from transformers import pipeline

# Load emotion classifier
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Example usage
text = "I feel a bit let down"
result = emotion_classifier(text)

print(result)
