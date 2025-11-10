# necessary library
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

#PART 1: URL CLASSIFIER
print("Training URL Classifier")

def calculate_entropy(s):
    """Calculate the Shannon entropy of a string."""
    if not s or not isinstance(s, str): return 0
    probabilities = [count / len(s) for _, count in Counter(s).most_common()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

TOP_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "instagram.com", "linkedin.com", "microsoft.com", "apple.com",
    "netflix.com", "paypal.com", "yahoo.com", "reddit.com", "office.com",
    "github.com", "bankofamerica.com"
}

def extract_features_final_robust(url):
    """
    Extracts all 24 features and robustly handles URLs missing a protocol.
    """
    #Normalize the URL
    # If a URL doesn't start with 'http', prepend 'http://' to it.
    if not re.match(r'^https?://', url):
        url = 'http://' + url

    try:
        features = []
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc if parsed_url.netloc else ''
        path = parsed_url.path if parsed_url.path else ''
        query = parsed_url.query if parsed_url.query else ''

        features.extend([len(url), len(hostname), len(path), len(query), len(parsed_url.fragment), url.count('-'), url.count('@'), url.count('?'), url.count('&'), url.count('='), url.count('.')])
        cleaned_hostname = hostname.replace('www.', '') if hostname.startswith('www.') else hostname
        features.append(cleaned_hostname.count('.'))
        features.append(1 if parsed_url.scheme == 'https' else 0)
        has_ip = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', hostname)
        features.append(1 if has_ip else 0)
        port = parsed_url.port
        features.extend([1 if port and port not in [80, 443] else 0, 1 if hostname.startswith('xn--') else 0, calculate_entropy(hostname), sum(c.isdigit() for c in hostname) if not has_ip else 0])
        brand_keywords = ['google', 'facebook', 'apple', 'paypal', 'amazon', 'microsoft', 'bank', 'ebay']
        features.append(1 if any(b in cleaned_hostname and not cleaned_hostname.startswith(b) for b in brand_keywords) else 0)
        shortener_keywords = ['bit.ly', 'goo.gl', 't.co', 'ow.ly', 'tinyurl']
        features.append(1 if any(sk in hostname for sk in shortener_keywords) else 0)
        parts = hostname.split('.')
        main_domain = ".".join(parts[-2:]) if len(parts) > 1 else hostname
        features.append(1 if main_domain in TOP_DOMAINS else 0)
        vowels = "aeiou"
        num_vowels = sum(1 for char in hostname if char in vowels)
        num_consonants = sum(1 for char in hostname if char.isalpha() and char not in vowels)
        features.append(num_vowels / (num_consonants + 1e-6))
        digit_sequences = re.findall(r'\d+', hostname)
        features.append(max(len(s) for s in digit_sequences) if digit_sequences else 0)
        consonant_sequences = re.findall(r'[^aeiou\d\W_]+', hostname, re.IGNORECASE)
        features.append(max(len(s) for s in consonant_sequences) if consonant_sequences else 0)
        if len(features) != 24: return [0] * 24
        return features

    except Exception:
        #The robust error handler for any other unexpected issues
        return [0] * 24

final_feature_names = [
    'url_length', 'hostname_length', 'path_length', 'query_length', 'fragment_length',
    'count_-', 'count_@', 'count_?', 'count_&', 'count_=', 'count_.',
    'num_subdomains', 'has_https', 'has_ip', 'has_uncommon_port',
    'has_punycode', 'hostname_entropy', 'digits_in_hostname', 'contains_deceptive_brand',
    'has_shortener', 'is_top_domain', 'vowel_consonant_ratio', 'longest_digit_seq', 'longest_consonant_seq'
]

#Load URL Data
print("Loading URL data from 'data/new_urls.csv'...")
url_df = pd.read_csv('data/new_urls.csv') 

#Extract the Final Feature Set
print("\nExtracting final set of 24 features...")
features_list = url_df['url'].apply(extract_features_final_robust).tolist()
features_df = pd.DataFrame(features_list, columns=final_feature_names)
print("Feature extraction complete.")

#Split the Data
X = features_df
y = url_df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nUsing ALL URL data for final training...")
X_train = X
y_train = y

# --- 4. Train a Stable RandomForestClassifier ---
print("\nTraining a stable RandomForestClassifier model...")
# We use strong default parameters that are known to work well.
model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete!")

print("URL Classifier Trained ---")


#EMAIL CLASSIFIER 
print("\n Training Email Classifier ---")

# --- 1. Load Email Data ---
print("Loading Email data from 'data/emails.csv'...")
# Changed Windows path to relative path
df = pd.read_csv("data/emails.csv")

df['label'] = df['Email Type'].map({'Safe Email':0,'Phishing Email':1})
X= df['Email Text']
y= df['label']

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size= 0.2, random_state = 42, stratify=y
)
print("\nUsing ALL Email data for final training...")
X_train = X
y_train = y

X_train = X_train.fillna("")


#converting to numbers
print("Vectorizing email text...")
vector = TfidfVectorizer(stop_words='english',max_features = 5000)
X_train_vec = vector.fit_transform(X_train)
# X_test_vec = vector.transform(X_test) # Not needed

#training
print("Training LogisticRegression model...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_vec, y_train)
print("Email model training complete!")

print("Email Classifier Trained ---")


#SAVING ALL MODELS TO 'models'
print("\n Saving All Models ")

# Create the new directory 
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)
print(f"Saving files to new directory: '{output_dir}'")

# Saving URL Model from url.ipynb
model_filename = os.path.join(output_dir, 'phishing_url_detector_model.joblib')
features_filename = os.path.join(output_dir, 'phishing_url_detector_features.joblib')

joblib.dump(model, model_filename)
print(f"URL Model saved successfully to '{model_filename}'")
joblib.dump(final_feature_names, features_filename)
print(f"URL Feature names saved successfully to '{features_filename}'")

#Saving Email Model from email_classifier.ipynb
email_model_filename = os.path.join(output_dir, 'email_classifier_model.joblib')
email_vectorizer_filename = os.path.join(output_dir, 'email_classifier_vectorizer.joblib')

joblib.dump(lr, email_model_filename)
print(f"Email Model saved successfully to '{email_model_filename}'")
joblib.dump(vector, email_vectorizer_filename)
print(f"Email Vectorizer saved successfully to '{email_vectorizer_filename}'")

print("\nAll new models saved!")