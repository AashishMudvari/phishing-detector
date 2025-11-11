Phishing Email and URL Detector



This project contains a machine learning system designed to detect phishing attempts with high accuracy. It analyzes both the text content of an email and the structural and lexical properties of any URLs found within it. The final combined model achieves an accuracy of approx. 91%.



Project Structure

The project is organized into a clean, easy-to-navigate folder structure:

phishing-detector/

├── data/

│   ├── emails.csv

│   └── new_urls.csv

│

├── phishnetweb.py

├── train.py

├── requirementsforubu.txt

├── requirements.txt

└── README.md



Files

-phishnetweb.py: The main, interactive script that loads the trained models and performs predictions on new emails provided by the user.

-data/: Contains the original `.csv` datasets used for training the models.


-requirements.txt: A list of all necessary Python libraries to run the project.

-README.md: This instruction file.



To get started, follow these steps:



1\. Set up the Environment

for ubuntu users
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
##making sure the device has all python, git and virtual environment packages 

Make sure you have Python 3 installed. Then, navigate to the project's root directory in your terminal and run the following command to install all required libraries:

To get started:
step 1: clone the repository

git clone https://github.com/AashishMudvari/phishing-detector.git

step2: create a virtual environment and activate(recommended)
For Linux/macOS
python3 -m venv venv
source venv/bin/activate

For Windows
python -m venv venv
.\venv\Scripts\activate

Step 3: install the requirements
For Ununtu
pip install -r requirementsforubu.txt

For Ununtu
pip install -r requirements.txt

Step 4: 
Use python (or python3 on some systems)
python train.py

This will create a models/ folder on your local machine containing the trained models.


Step5: Run the Application Now you can start the web app:

streamlit run phishnetweb.py


Models

This system uses two distinct models that work together for a final prediction:

URL Classifier: A RandomForestClassifier trained on over 800,000 URLs. It analyzes 24 structural and lexical features of a URL to determine if it's malicious.

Email Classifier: A LogisticRegression model trained on a dataset of emails. It uses TF-IDF vectorization to analyze the text content for keywords and patterns common in phishing attacks.

