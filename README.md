# Tom Nook's Prediction API

This FastAPI backend serves as the core intelligence engine for the Prediction Shop. It manages multiple machine learning models trained on Animal Crossing: New Horizons datasets to provide real-time game-play insights.

## 🚀 Tech Stack
- **Framework:** FastAPI
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Asynchronous Execution:** Uvicorn
- **Deployment:** Render / Google Colab (Development)

## 🛠️ Installation & Setup

1. **Navigate to the backend directory:**
   ```bash
   cd tom-nook-prediction-backend
   ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
   
3. **Install requirements:**
    ```bash
   pip install -r requirements.txt
    ```
   
4. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```