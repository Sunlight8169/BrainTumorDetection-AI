#  Brain Tumor Detection & Medical Assistance System

This project is an AI-powered Brain Tumor Detection System designed to analyze MRI images using a CNN deep learning model.
It also includes an intelligent NLP-based Medical Chatbot to guide users by answering brain tumor–related queries and providing awareness.

##  This system aims to support radiologists & hospitals by enabling:
✔ Faster initial screening
✔ Reduced diagnostic effort
✔ Better medical decision support

##  Features
Feature	Description
MRI Image Classification	Detects 4 types of brain conditions (Glioma / Meningioma / Pituitary / No Tumor)
Deep Learning Model	Trained CNN using TensorFlow & Keras
Streamlit Frontend	User can upload MRI image easily
Flask Backend	Handles model prediction & report generation
PDF Report Generator	Exports medical-style tumor report
Medical Chatbot	Answers tumor-related queries using sentence-transformers

##  Project File Overview
File Name	Purpose
main.py	Flask backend API for tumor prediction & report generation
streamlit_app.py	Streamlit UI (MRI upload + Prediction + Chatbot UI)
saved_model.h5	Final trained CNN model file
chatbot_engine.py	Main chatbot workflow logic
nlp_processor.py	Handles NLP embeddings and similarity search
knowledge_base_manager.py	Stores medical Q/A knowledge database
braintumor-ipynb (2).ipynb	Model training & evaluation notebook
README.md	Documentation of the project
 Model Details

Model Type: Convolutional Neural Network (CNN)

Classes:

Glioma

Meningioma

Pituitary

No Tumor

## Achieved Results:

Training Accuracy: ~90%

Validation Accuracy: ~85%

Test Accuracy: ~67%

🛠 Technologies Used
Category	Tools / Libraries
Programming	Python 3.10
Deep Learning	TensorFlow, Keras
Image Processing	PIL / OpenCV
Data Handling	NumPy, Pandas
Frontend	Streamlit
Backend	Flask, Flask-CORS
NLP	Sentence-Transformers
Deployment	GitHub
📷 System Workflow (High-Level)
User MRI Input → Preprocessing → CNN Model Prediction
→ Tumor Type Display → PDF Report
→ Medical Chatbot Assistance

How to Run Locally
pip install -r requirements.txt
cd backend
python main.py   (Start Backend)
cd ../chatbot
streamlit run streamlit_app.py   (Start Frontend)


Backend Runs On → http://127.0.0.1:5000
Frontend Runs On → http://localhost:8501


###  DataSet Link -- https://www.kaggle.com/datasets/dadavishwakarma/braintumor
