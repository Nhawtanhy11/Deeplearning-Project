import pickle
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from transformers import AlbertTokenizer, AlbertForSequenceClassification, BertTokenizer, XLNetTokenizer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import transformers

# Initialize Flask app
app = Flask(__name__)

# Define sentiment labels
labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

# Load Albert model and tokenizer
albert_model_path = "C:\\Users\\admin\\Downloads\\Albert\\model"
albert_tokenizer_path = "C:\\Users\\admin\\Downloads\\Albert\\tokenizer"

try:
    albert_model = AlbertForSequenceClassification.from_pretrained(albert_model_path)
    albert_tokenizer = AlbertTokenizer.from_pretrained(albert_tokenizer_path)
    albert_model.eval()  # Set to evaluation mode
except Exception as e:
    print(f"Error loading Albert model or tokenizer: {e}")

# Load BERT-based Sentiment Classifier
MODEL = "bert-base-uncased"
NUM_LABELS = 5
bert_model_path = "C:\\Users\\admin\\Downloads\\save_model.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.7):
        super(SentimentClassifier, self).__init__()
        self.config = transformers.BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = transformers.BertModel.from_pretrained(model_name, config=self.config)
        self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

try:
    bert_model = SentimentClassifier(MODEL, NUM_LABELS).to(device)
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=device))
    bert_model.eval()  # Set to evaluation mode
    bert_tokenizer = BertTokenizer.from_pretrained(MODEL)
except Exception as e:
    print(f"Error loading BERT model or tokenizer: {e}")
    
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

class XLNetWithDropout(nn.Module):
    def __init__(self, model_name, num_labels):
        super(XLNetWithDropout, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.4)
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classifier(self.dropout(pooled_output))
        return {"logits": logits}
    
xlnet_model_path = "C:\\Users\\admin\\Downloads\\xlnet_sst5_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlnet_model = XLNetWithDropout("xlnet-base-cased", num_labels=5).to(device)
xlnet_model.load_state_dict(torch.load(xlnet_model_path, map_location=device))
xlnet_model.eval()

def get_model_sentiment_xlnet(model, tokenizer, text, device):
    # Tokenize input text
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().flatten()
        predicted_class = logits.argmax(dim=-1).item()

    # Map to sentiment label
    sentiment = labels[predicted_class]
    return sentiment, probabilities

class DebertaV3WithDropout(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, dropout_prob=0.2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=num_labels
        )
        self.dropout = nn.Dropout(dropout_prob)
        hidden_size = self.model.config.hidden_size
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # New classification head with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_labels)
        )
        self.model.classifier = self.classifier

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.dropout(outputs.logits)
        return {"loss": outputs.loss, "logits": logits} if labels is not None else {"logits": logits}

def deberta_load_model(model_path, model_class, tokenizer_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = model_class(pretrained_model_name=tokenizer_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer

# Prediction function
def get_model_sentiment_deberta(text, model, tokenizer, max_length=128):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    inputs.pop('token_type_ids', None)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        predicted_label = torch.argmax(logits, dim=-1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().tolist()

    return predicted_label, probabilities

deberta_model_path = "C:\\Users\\admin\\Downloads\\deberta_model_checkpoint (1).pth"
deberta_model, deberta_tokenizer = deberta_load_model(deberta_model_path, DebertaV3WithDropout, "microsoft/deberta-v3-base", NUM_LABELS)

# Keras Model Loading (CNN and LSTM)
# Function to load the tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer

# Function to load the model
def keras_load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def preprocess_text(text, tokenizer, max_length=50):
    # Convert text to sequence of integers based on the tokenizer
    sequence = tokenizer.texts_to_sequences([text])
    # Pad sequences to ensure uniform length
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

label_encoder = LabelEncoder()
label_encoder.fit([0, 1, 2, 3, 4])

# Function to make predictions
def predict(model, tokenizer, text, max_length=32):
    processed_text = preprocess_text(text, tokenizer, max_length)
    predictions = model.predict(processed_text)
    probabilities = tf.nn.softmax(predictions, axis=-1).numpy().flatten()
    predicted_class = predictions.argmax(axis=-1)
    decoded_label = label_encoder.inverse_transform([predicted_class[0]])[0]
    return decoded_label, probabilities

# Paths to saved CNN and LSTM models and tokenizers
CNN_model_path = "C:\\Users\\admin\\Downloads\\CNN\\CNN.keras"  
CNN_tokenizer_path = "C:\\Users\\admin\\Downloads\\CNN\\tokenizer.pkl"  
LSTM_model_path = "C:\\Users\\admin\\Downloads\\LSTM\\LSTM_model.keras"  
LSTM_tokenizer_path = "C:\\Users\\admin\\Downloads\\LSTM\\LSTM_tokenizer.pkl"  

# Load CNN and LSTM models
cnn_model = keras_load_model(CNN_model_path)
cnn_tokenizer = load_tokenizer(CNN_tokenizer_path)
lstm_model = keras_load_model(LSTM_model_path)
lstm_tokenizer = load_tokenizer(LSTM_tokenizer_path)

# Generalized sentiment prediction function
def get_model_sentiment(model, tokenizer, text):
    try:
        if isinstance(model, AlbertForSequenceClassification):
            # Albert processing
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
        else:
            # BERT processing
            encoding = tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=512, return_attention_mask=True,
                return_tensors='pt', padding='max_length', truncation=True
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
        prediction = torch.argmax(logits, dim=-1).item()
        return labels[prediction], probabilities
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return "Error", []

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in 'templates/'

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment_response():
    try:
        user_message = request.json.get("message", "")
        
        # Get predictions from all models
        albert_prediction, albert_probabilities = get_model_sentiment(albert_model, albert_tokenizer, user_message)
        bert_prediction, bert_probabilities = get_model_sentiment(bert_model, bert_tokenizer, user_message)
        cnn_prediction, cnn_probabilities = predict(cnn_model, cnn_tokenizer, user_message)
        lstm_prediction, lstm_probabilities = predict(lstm_model, lstm_tokenizer, user_message)
        xlnet_prediction, xlnet_probabilities = get_model_sentiment_xlnet(xlnet_model, xlnet_tokenizer, user_message, device)
        deberta_prediction, deberta_probabilities = get_model_sentiment_deberta(user_message, deberta_model, deberta_tokenizer)
        

        # Return predictions from all models
        return jsonify({
            "models": {
                "Albert": {
                    "prediction": albert_prediction,
                    "probabilities": {k: float(v) for k, v in zip(labels, albert_probabilities)}
                },
                "BERT": {
                    "prediction": bert_prediction,
                    "probabilities": {k: float(v) for k, v in zip(labels, bert_probabilities)}
                },
                "XLNet": {
                    "prediction": xlnet_prediction,
                    "probabilities": {k: float(v) for k, v in zip(labels, xlnet_probabilities)}
                },
                "Deberta": {
                    "prediction": labels[int(deberta_prediction)],
                    "probabilities": {k: float(v) for k, v in zip(labels, deberta_probabilities[0])}
                },
                "CNN": {
                    "prediction": labels[int(cnn_prediction)],
                    "probabilities": {k: float(v) for k, v in zip(labels, cnn_probabilities)}
                },
                "LSTM": {
                    "prediction": labels[int(lstm_prediction)],
                    "probabilities": {k: float(v) for k, v in zip(labels, lstm_probabilities)}
                }
            }
        })
    except Exception as e:
        print(f"Error in API: {e}")
        return jsonify({"error": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
