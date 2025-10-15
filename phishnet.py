import joblib
import pandas as pd
import re
from urllib.parse import urlparse
from collections import Counter
import math

#final feature extraction
def calculate_entropy(s):
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

def final_extract_feature(url):
    """Extracts all 24 features and handles missing protocol."""
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
        return [0] * 24


try:
    # --- URL Model Components ---
    url_model = joblib.load('models/phishing_url_detector_model.joblib')
    url_feature_names = joblib.load('models/phishing_url_detector_features.joblib')

    # --- Email Model Components ---
    email_model = joblib.load('models/email_classifier_model.joblib')
    email_vectorizer = joblib.load('models/email_classifier_vectorizer.joblib')
    
    print("models and components loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find one or more model files. issue with location of joblib files.")
    exit()

#combining the models
def predict_phishing(email_body, url_weight=0.6, email_weight=0.4):
    
    #Analysis of email body, extracts the first url, and returns a combined, weighted phishing score
    
    url_match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', email_body)#url analysis
    
    if url_match:
        url = url_match.group(0)
        print(f"URL Found: {url}")
        
        #phishing probability
        url_features = final_extract_feature(url)
        url_features_df = pd.DataFrame([url_features], columns=url_feature_names)
        p_url = url_model.predict_proba(url_features_df)[0][0] # P(phishing) is at index 0 for RandomForest
    else:
        print("No URL found in the email. URL score will be 0.")
        p_url = 0.0 #no URL and based risk

    email_vec = email_vectorizer.transform([email_body])
    p_email = email_model.predict_proba(email_vec)[0][1] # P(phishing) is at index 1 for LogisticRegression

    print(f"URL Model Score (Probability of phishing): {p_url:.2%}")
    print(f"Email Text Score (Probability of phishing): {p_email:.2%}")
    
    # Calculate the weighted average
    final_score = (p_url * url_weight) + (p_email * email_weight)
    
    return final_score

#example
if __name__ == '__main__':
    # A tricky email that relies on both a deceptive URL and suspicious text
    sample_phishing_email = """
    Subject: Urgent: Your account requires verification

    Dear Valued Client,

    We have detected suspicious activity on your account from an unrecognized device. 
    For your security, access has been temporarily restricted.

    To restore access, you must verify your identity immediately. 
    Please click the link below to proceed with the verification process.

    Verify Now: http://microsaft-secure-login.com/auth/update

    Failure to do so within 24 hours will result in permanent account suspension.

    Thank you,
    The Security Department
    """

    print("Analyzing a sample email ")
    final_phishing_score = predict_phishing(sample_phishing_email)
    final_decision = "Likely Phishing" if final_phishing_score > 0.5 else "Likely Legitimate"

    print("\n" + "_"*40)
    print(f"Combined Phishing Score: {final_phishing_score:.2%}")
    print(f"Final Decision: {final_decision}")
    print("_"*40)
#asking user for a sample email
if __name__ == '__main__':
    print("\nPhishing Email Detector ")
    print("Enter the body of an email to analyze. Type 'exit' to quit.")

    while True:
        # Prompt the user to enter email text
        user_input = input("\nPaste email text here:\n> ")

        if user_input.lower() == 'exit':
            break

        if not user_input.strip():
            print("Input is empty. Please paste some text.")
            continue

        # Analyze the user's email
        final_phishing_score = predict_phishing(user_input)
        final_decision = "Likely Phishing " if final_phishing_score > 0.5 else "Likely Legitimate "

        print("\n" + "_"*40)
        print(f"Combined Phishing Score: {final_phishing_score:.2%}")
        print(f"Final Decision: {final_decision}")
        print("_"*40)