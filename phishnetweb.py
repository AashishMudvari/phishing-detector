import streamlit as st
import joblib
import pandas as pd
import re
from urllib.parse import urlparse
from collections import Counter
import math

#loading the component
@st.cache_resource#caching for performance
def load_models():
    try:
        # URL Model Components
        url_model = joblib.load('models/phishing_url_detector_model.joblib')
        url_feature_names = joblib.load('models/phishing_url_detector_features.joblib')

        # Email Model Components
        email_model = joblib.load('models/email_classifier_model.joblib')
        email_vectorizer = joblib.load('models/email_classifier_vectorizer.joblib')
        
        return url_model, url_feature_names, email_model, email_vectorizer
    except FileNotFoundError:
        st.error("Error: Model files not found. Ensure they are in a 'models/' sub-folder.")
        return None, None, None, None

# Load everything
url_model, url_feature_names, email_model, email_vectorizer = load_models()


def calculate_entropy(s): #feature extraction
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
    #extracts all 24 features and handles URLs missing a protocol
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

#logic
def predict_phishing(email_body, url_weight=0.6, email_weight=0.4):
    url_match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', email_body)
    
    p_url = 0.0
    url_found = "No URL found in the email, this will likely flag the email safe, but be cautious with request of information. "
    if url_match:
        url = url_match.group(0)
        url_found = f"URL Found: {url}"
        url_features = extract_features_final_robust(url)
        url_features_df = pd.DataFrame([url_features], columns=url_feature_names)
        p_url = url_model.predict_proba(url_features_df)[0][0]

    email_vec = email_vectorizer.transform([email_body])
    p_email = email_model.predict_proba(email_vec)[0][1]

    final_score = (p_url * url_weight) + (p_email * email_weight)
    
    return final_score, p_url, p_email, url_found

#
st.set_page_config(page_title="Phishing Detector", layout="wide")
st.title("Phishing Email Detector")

#column for text box
email_text = st.text_area("Paste the email content here:", height=300)
analyze_button = st.button("Scan")


#analysis and result
if analyze_button and email_text:
    if all([url_model, url_feature_names, email_model, email_vectorizer]):
        with st.spinner('Analyzing...'):
            final_score, p_url, p_email, url_found_text = predict_phishing(email_text)
            
            st.markdown("---")
            st.header("Analysis Results")
            
            # Display individual model scores
            st.info(url_found_text)
            st.write(f"URL Model Score (Probability of phishing):`{p_url:.2%}`")
            st.write(f"Email Text Score (Probability of phishing):`{p_email:.2%}`")
            
            # Display final decision
            if final_score > 0.7:
                st.error(f"Final Score: {final_score:.2%} - High Risk ")
                st.warning("This email is very likely a phishing attempt. Do not click any links or provide any information.")
            elif final_score > 0.5:
                st.warning(f"Final Score: {final_score:.2%} - Caution Advised ")
                st.info("This email shows some signs of phishing. Please be cautious with any links or requests for information.")
            else:
                st.success(f"**Final Score: {final_score:.2%} - Likely Safe ")
                st.info("This email appears to be safe, but always remain vigilant.")
    else:
        st.error("Cannot perform analysis because model files were not loaded correctly.")
elif analyze_button:
    st.warning("Please paste some email content to analyze.")