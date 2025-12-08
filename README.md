🏥 HealthE – AI-Powered Wellness & Recovery Assistant

HealthE is an AI-powered health companion that predicts recovery time from illness and recommends personalized diet plans using two machine-learning models.
Built as a full-stack application with:

Streamlit Frontend

FastAPI Backend

PostgreSQL (NeonDB) Database

Docker & Docker Compose for complete containerization

Joblib-based ML Models (LightGBM + Logistic Regression)

This project was created as part of a capstone by a team of 6 students.

🚀 Features
🔧 Recovery Days Prediction

Uses a trained LightGBM regression model + preprocessing scaler to estimate expected recovery time.

🍎 Diet Plan Recommendation

Uses a trained Logistic Regression classifier to recommend one of several diet plans based on lifestyle + health metrics.

📊 History Tracking (Database Logging)

Every prediction is saved to PostgreSQL, allowing the team to view the latest 10 entries inside the sidebar UI.

🐳 Full Dockerized Architecture

Using Docker Compose, the complete system runs with one command:

docker compose up --build

📁 Clean Modular Folder Structure
HealthE/
│── app.py                     # Streamlit frontend
│── backend/
│     └── main.py              # FastAPI backend + DB models
│── models/
│     └── *.joblib             # ML models & scalers
│── data/
│     └── *.csv                # Original datasets
│── requirements.txt           # Python dependencies
│── docker-compose.yml
│── Dockerfile                 # Single Dockerfile used for both services
│── .env                       # Environment variables (DB URL, Backend URL)
└── README.md

🛠️ Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	FastAPI
ML	LightGBM, Scikit-Learn, Joblib
Database	PostgreSQL (Neon)
Orchestration	Docker Compose
Language	Python 3.10
⚙️ Running the Project
✅ Setup .env

Inside your project root:

# Backend
DB_URL=postgresql://<user>:<password>@<host>/<database>
BACKEND_URL=http://backend:8000

# Frontend
STREAMLIT_SERVER_PORT=8501

🐳 Run everything with Docker Compose
docker compose up --build


Services will start:

Service	URL
Frontend (Streamlit)	http://localhost:8501

Backend (FastAPI)	http://localhost:8000/docs
🔌 API Endpoints (FastAPI)
➕ POST /log_prediction

Stores prediction + input features in DB.

📜 GET /history?limit=10

Returns the 10 most recent predictions saved.

All endpoints are visible in Swagger UI:

👉 http://localhost:8000/docs

🤖 ML Models
1️⃣ Recovery Time Model

Algorithm: LightGBM Regressor

Input features: Age, BMI, condition, sleep hours, smoking status, severity, etc.

Output: Predicted recovery days

2️⃣ Diet Recommendation Model

Algorithm: Logistic Regression

Output Categories:

Balanced Diet

High Protein

Keto

Low Carb

Low Fat

Vegan

Both models are pre-trained and stored in models/*.joblib.

📊 Data Logging & History

Every prediction sent from the UI is logged to PostgreSQL via backend FastAPI.

Example stored entry:

{
  "prediction_type": "recovery_days",
  "inputs": { ... },
  "output": { "recovery_days": 7.5 },
  "created_at": "2025-02-05T18:40:31"
}

📸 Screenshots

(Add later)

🟩 Streamlit UI  
🟦 FastAPI Docs  
🟪 Database with stored predictions  

🧪 Testing (Optional)

Run backend only:

uvicorn backend.main:app --reload


Run frontend only:

streamlit run app.py

🧑‍🤝‍🧑 Project Contributors

💡 Developed by 6 students from Durham College
⚙️ ML, backend, frontend, and DevOps contributions distributed across the team.

⭐ If you like this project, give it a star!

This helps the team demonstrate good engineering practices, containerization, and full-stack ML deployment.