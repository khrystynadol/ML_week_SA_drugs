import torch
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification

# Define checkpoint path (Change this to your checkpoint path)
checkpoint_path = "checkpoint-27705"

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(checkpoint_path)

# Load model from checkpoint
model = BertForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

# JSON file path to store drug reviews
json_file_path = r"data\drug_reviews_grouped.json"

def map_label(predicted_label):
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_mapping.get(predicted_label, "Unknown")

def predict_label(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get prediction
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

def add_review(drug_name, user_review):
    # Predict the label for the given review
    predicted_label = predict_label(user_review)
    mapped_label = map_label(predicted_label)

    # Print prediction result
    print(f"Drug name = {drug_name}; Predicted Label: {mapped_label}")

    # Only add to dataset if the review is labeled as "Negative"
    if mapped_label == "Negative":
        # Load existing JSON data (or create a new structure if file doesn't exist)
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                try:
                    data = json.load(json_file)
                except json.JSONDecodeError:
                    data = {}  # If file is empty or corrupted, start fresh
        else:
            data = {}

        # Ensure the drug has an entry in the dataset
        if drug_name not in data:
            data[drug_name] = {"review": []}

        # Append the negative review
        data[drug_name]["review"].append(user_review)

        # Save updated JSON back to file
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Negative review added successfully to {json_file_path}")
    else:
        print("Review is not negative. It will not be added to the dataset.")

# Example Usage
review = "It worked really bad, my condition got worse, I had a headache because of these pills"
drug_name = "Paracetamol"
add_review(drug_name, review)

review_1 = "I felt such a relief, it felt amazing after all this pain"
drug_name_1 = "Paracetamol"
add_review(drug_name_1, review_1)