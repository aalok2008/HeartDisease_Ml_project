import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and columns
def load_artifacts():
    model = joblib.load('KNN_heart.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# Categorical options (from the original dataset)
sex_options = ['M', 'F']
chest_pain_options = ['ATA', 'NAP', 'ASY', 'TA']
resting_ecg_options = ['Normal', 'ST', 'LVH']
exercise_angina_options = ['Y', 'N']
st_slope_options = ['Up', 'Flat', 'Down']

# Preprocessing function to match training encoding
def preprocess_input(
    Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
):
    input_dict = {
        'Age': [Age],
        'Sex': [Sex],
        'ChestPainType': [ChestPainType],
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'FastingBS': [FastingBS],
        'RestingECG': [RestingECG],
        'MaxHR': [MaxHR],
        'ExerciseAngina': [ExerciseAngina],
        'Oldpeak': [Oldpeak],
        'ST_Slope': [ST_Slope],
    }
    df = pd.DataFrame(input_dict)
    df_encoded = pd.get_dummies(df, drop_first=True)
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[columns]
    X_scaled = scaler.transform(df_encoded)
    return X_scaled

def predict(
    Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
):
    X = preprocess_input(
        Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
        RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    )
    pred = model.predict(X)[0]
    return """
    <div class='alert {}'>
        <span class='alert-icon'>{}</span>
        <span class='alert-text'>Heart Disease: <b>{}</b></span>
    </div>
    """.format(
        'alert-danger' if pred == 1 else 'alert-success',
        '⚠️' if pred == 1 else '✅',
        'Yes' if pred == 1 else 'No'
    )

def build_ui():
    custom_css = """
    body, .gradio-container {background: #18181b !important;}
    .main-card {background: #232329; border-radius: 18px; box-shadow: 0 4px 32px #0008; padding: 2.5rem 2rem; max-width: 650px; margin: 2rem auto; border: 1px solid #23232f;}
    .alert {padding: 1.2rem 1rem; border-radius: 10px; font-weight: 600; margin-top: 2rem; text-align: center; font-size: 1.25rem; display: flex; align-items: center; justify-content: center; gap: 0.7em;}
    .alert-success {background: #1a2e22; color: #6ee7b7; border: 1.5px solid #22c55e;}
    .alert-danger {background: #2a1a1a; color: #f87171; border: 1.5px solid #ef4444;}
    .alert-icon {font-size: 1.7em;}
    .alert-text {font-size: 1.15em;}
    .footer {margin-top: 2.5rem; text-align: center; color: #888; font-size: 1rem;}
    .logo-heart {font-size:2.2em; color:#ef4444; vertical-align:middle; margin-right:0.2em;}
    .gr-button {background: linear-gradient(90deg,#22d3ee,#818cf8); color: #fff; border-radius: 8px; font-weight: 600; font-size: 1.1em; border: none; transition: box-shadow 0.2s; box-shadow: 0 2px 8px #0003;}
    .gr-button:hover {box-shadow: 0 4px 16px #22d3ee55, 0 2px 8px #0003;}
    .gr-input, .gr-dropdown, .gr-number, .gr-radio {background: #232329; color: #fff; border-radius: 6px; border: 1px solid #333;}
    .gr-input:focus, .gr-dropdown:focus, .gr-number:focus, .gr-radio:focus {border: 1.5px solid #818cf8;}
    label {color: #cbd5e1 !important;}
    #section-header {color: #a5b4fc; font-size: 1.2em; margin-bottom: 1em;}
    """
    with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
        gr.Markdown("""
        <div style='text-align:center; margin-bottom:1.5rem;'>
            <span class='logo-heart'>❤️</span>
            <h1 style='margin-bottom:0.2em; display:inline; color:#fff;'>Heart Disease Prediction</h1>
            <div style='color:#a5b4fc; font-size:1.1em;'>KNN Model &middot; Dark UI</div>
        </div>
        """)
        with gr.Column(elem_classes="main-card"):
            gr.Markdown("### Patient Information", elem_id="section-header")
            with gr.Row():
                Age = gr.Number(label="Age", value=50, info="Enter age in years", interactive=True)
                Sex = gr.Dropdown(choices=sex_options, label="Sex", value="M", info="Biological sex (M/F)", interactive=True)
            with gr.Row():
                ChestPainType = gr.Dropdown(choices=chest_pain_options, label="Chest Pain Type", value="ATA", info="Type of chest pain (e.g. ATA, NAP)", interactive=True)
                RestingBP = gr.Number(label="Resting Blood Pressure", value=120, info="mm Hg", interactive=True)
            with gr.Row():
                Cholesterol = gr.Number(label="Cholesterol", value=200, info="mg/dl", interactive=True)
                FastingBS = gr.Radio(choices=[0, 1], label="Fasting Blood Sugar > 120 mg/dl", value=0, info="1 = True, 0 = False", interactive=True)
            with gr.Row():
                RestingECG = gr.Dropdown(choices=resting_ecg_options, label="Resting ECG", value="Normal", info="ECG results", interactive=True)
                MaxHR = gr.Number(label="Max Heart Rate", value=150, info="Maximum heart rate achieved", interactive=True)
            with gr.Row():
                ExerciseAngina = gr.Dropdown(choices=exercise_angina_options, label="Exercise Induced Angina", value="N", info="Y = Yes, N = No", interactive=True)
                Oldpeak = gr.Number(label="Oldpeak (ST depression)", value=1.0, info="ST depression induced by exercise", interactive=True)
            with gr.Row():
                ST_Slope = gr.Dropdown(choices=st_slope_options, label="ST Slope", value="Up", info="Slope of the peak exercise ST segment", interactive=True)
            submit_btn = gr.Button("🔍 Predict", elem_id="predict-btn", scale=2)
            output = gr.HTML(label="Prediction")
            submit_btn.click(
                predict,
                inputs=[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope],
                outputs=output
            )
        gr.Markdown("""
        <div class='footer'>
            Made with <span style='color:#e25555;'>&#9829;</span> using Gradio &middot; UI inspired by <a href='https://ui.shadcn.com/' target='_blank' style='color:#818cf8;'>shadcn/ui</a>
        </div>
        """)
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch() 