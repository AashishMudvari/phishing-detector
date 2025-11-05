Phishing Email and URL Detector



This project contains a machine learning system designed to detect phishing attempts with high accuracy. It analyzes both the text content of an email and the structural and lexical properties of any URLs found within it. The final combined model achieves an accuracy of approx. 91%.



Project Structure

The project is organized into a clean, easy-to-navigate folder structure:

phishing-detector/

├── data/

│   ├── emails.csv

│   └── urls.csv

│

├── models/

│   ├── email\_classifier\_model.joblib

│   ├── email\_classifier\_vectorizer.joblib

│   ├── phishing\_url\_detector\_features.joblib

│   ├── phishing\_url\_detector\_model.joblib

│

├── phishnetweb.py

├── requirements.txt

└── README.md



Files

-phishnetweb.py: The main, interactive script that loads the trained models and performs predictions on new emails provided by the user.

-data/: Contains the original `.csv` datasets used for training the models.

-models/`\*\*: Contains the four final, saved model files (`.joblib`) required for prediction.

-requirements.txt: A list of all necessary Python libraries to run the project.

-README.md: This instruction file.



To get started, follow these steps:



1\. Set up the Environment

for ubuntu users
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
##making sure the device has all python, git and virtual environment packages 

Make sure you have Python 3 installed. Then, navigate to the project's root directory in your terminal and run the following command to install all required libraries:

create a virtual environment(recommended)
# Create a virtual environment named 'venv' inside your project folder
python3 -m venv venv

# Activate the environment
source venv/bin/activate



pip install -r requirementsforubu.txt


2\. Run the phishnetweb.py script

streamlit run phishnetweb.py


3\. Test an email

the script has a sample email to show the results and prompts user to paste a email to detect it. 

Type 'exit' to stop the program



**Models**

This system uses two distinct models that work together for a final prediction:



URL Classifier: A RandomForestClassifier trained on over 800,000 URLs. It analyzes 24 structural and lexical features of a URL to determine if it's malicious.



Email Classifier: A LogisticRegression model trained on a dataset of emails. It uses TF-IDF vectorization to analyze the text content for keywords and patterns common in phishing attacks.

