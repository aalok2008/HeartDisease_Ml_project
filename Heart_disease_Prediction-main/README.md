# Heart Disease Prediction

A machine-learning project that predicts the likelihood of heart disease from patient medical attributes using a pre-trained **K-Nearest Neighbors (KNN)** model.

## Project Structure

- `app.py` — Flask application that serves the prediction interface and endpoint.
- `Heart.ipynb` — Jupyter Notebook for exploratory data analysis, model training, and evaluation.
- `KNN_heart.pkl` — Pre-trained KNN model.
- `scaler.pkl` — Scaler used to preprocess input features.
- `columns.pkl` — Feature/column names expected by the model.
- `requirements.txt` — Python dependencies.
- `environment.yml` — Conda environment specification.

## Run Locally on Windows

1. **Download or clone this repository:**

   ```bash
   git clone https://github.com/aalok2008/Heart_disease_Prediction.git
   cd Heart_disease_Prediction
   ```

   If you have downloaded the ZIP instead, extract it and open the extracted project folder in VS Code.

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the environment in Windows PowerShell:**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, you can use Command Prompt and run:

   ```bat
   venv\Scripts\activate.bat
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Start the application:**

   ```bash
   python app.py
   ```

6. Open `http://127.0.0.1:5000` in your browser if the application starts on that address.

## GitHub

GitHub profile: https://github.com/aalok2008

Repository URL (after you create and publish it):  
https://github.com/aalok2008/Heart_disease_Prediction

## Live Demo

No live demo link is configured in this README yet. Add your own deployment URL here after you deploy the application.

## Note

This project is for educational purposes only and is not a substitute for professional medical advice or diagnosis.
