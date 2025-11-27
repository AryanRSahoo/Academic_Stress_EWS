# Academic Stress Early Warning System (EWS)

A machine-learning-based Early Warning System designed to predict **academic stress risk levels** among high-school students using behavioural, demographic, and academic indicators.

This project includes:

- A **desktop application** (Tkinter)
- A **web application** (Streamlit)
- A trained **Logistic Regression prediction model**
- A clear **educational dataset sample**
- Full, open-source implementation for transparency and reproducibility

---

## 🚀 Features

### ✔ Predicts academic stress using 32 validated student features  
### ✔ Clean and user-friendly UI (both desktop & web)  
### ✔ Fully open-source and reproducible  
### ✔ Model trained using Python, scikit-learn, and pandas  
### ✔ Sample dataset included for demonstration  
### ✔ Designed as part of an academic ML research project  

---

## 📁 Project Structure
Academic_Stress_EWS/
│
├── models/
│ ├── logistic_pipeline.joblib # Trained ML model
│ └── feature_names.json # Ordered list of features
│
├── src/
│ ├── desktop/
│ │ └── app.py # Tkinter desktop app
│ ├── streamlit/
│ │ └── app.py # Streamlit web app
│
├── data/
│ └── student_stress_dataset_sample.csv # 15-row sample dataset (safe)
│
├── requirements-desktop.txt # Dependencies for desktop version
├── requirements-streamlit.txt # Dependencies for web version
└── README.md


---

## 🧠 Model Details

- **Model type:** Logistic Regression Pipeline  
- **Framework:** scikit-learn  
- **Training dataset:** Student performance + behavioral factors  
- **Target variable:** Stress risk (Low / Medium / High)

---

## 🖥 Running the Desktop App (Tkinter)

### **1. Create and activate environment**
python3 -m venv venv_desktop
source venv_desktop/bin/activate
### **2. Install dependencies
pip install -r requirements-desktop.txt
### **3. Run the app
python src/desktop/app.py

## 🌐 Running the Web App (Streamlit)

### **1. Create environment
python3 -m venv venv_streamlit
source venv_streamlit/bin/activate
### **2. Install
pip install -r requirements-streamlit.txt
### **3. Launch
streamlit run src/streamlit/app.py

## 📊 Dataset
A safe synthetic sample dataset (student_stress_dataset_sample.csv) with 32 features is provided for demonstration purposes.
The full dataset is not included for privacy reasons.

## 💡 Purpose
This project was developed as part of a high-impact academic research effort focusing on predicting and mitigating academic stress among students.
It serves as both an educational tool and a demonstration of applied machine learning.

## 📜 License
This project is open-source and distributed under the MIT License.
Feel free to use it for research, education, or extensions.

## 👤 Author
**Aryan Ryan Sahoo**
Email: aryansahoouni@gmail.com
GitHub: github.com/AryanRSahoo

## ⭐ Support the Project
If you find this project helpful, please consider giving it a star ⭐ on GitHub!
