import os
import json
import joblib
import pandas as pd
import streamlit as st
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import google.generativeai as genai

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

# --------- Helper: Preprocess inputs ----------
def prepare_input(form_data: Dict[str, Any], example_schema: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for k in example_schema.keys():
        row[k] = form_data.get(k, example_schema[k])
    return pd.DataFrame([row])

# --------- Gemini Chat wrapper ----------
def gemini_chat_reply(system_prompt: str, messages: list, gemini_api_key: str, model_name="gemini-1.5-flash"):
    """
    Mengirim percakapan ke model Gemini di Google AI Studio dan mengembalikan balasan teks.
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name)

    # Gabungkan semua percakapan menjadi satu teks panjang
    full_prompt = system_prompt + "\n\n"
    for m in messages:
        role = m["role"].upper()
        full_prompt += f"{role}: {m['content']}\n"

    # Panggil API Gemini
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memanggil Gemini API: {e}"

# --------- Streamlit App ----------
st.set_page_config(page_title="Prediksi Harga Mobil + Gemini AI", layout="wide")
st.title("🚗 Prediksi Harga Mobil & Chat Gemini AI")

# Sidebar: konfigurasi
st.sidebar.header("Konfigurasi")
model_path = st.sidebar.text_input("Path ke model (.pkl)", value="model.pkl")
example_json_path = st.sidebar.text_input("Path contoh fitur (example_features.json)", value="example_features.json")

gemini_key_env = st.sidebar.text_input("Gemini API Key (wajib)", value="", type="password")
use_env = st.sidebar.checkbox("Ambil dari ENV var GOOGLE_API_KEY jika kosong", value=True)

if use_env and not gemini_key_env:
    gemini_key_env = os.environ.get("GOOGLE_API_KEY", "")

# Load contoh schema
example_schema = {}
try:
    with open(example_json_path, "r") as f:
        example_schema = json.load(f)
except Exception as e:
    st.sidebar.warning(f"Gagal load example_features.json: {e}. Gunakan schema default.")
    example_schema = {
        "model": "Aygo",
        "year": 2017,
        "transmission": "Manual",
        "mileage": 11730,
        "fuelType": "Petrol",
        "tax": 0,
        "mpg": 68.9,
        "engineSize": 1.0
    }

# Load model prediksi
try:
    model = load_model(model_path)
    st.sidebar.success("Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model dari {model_path}: {e}")
    st.stop()

# Layout dua kolom
left, right = st.columns([1.2, 1.0])

# ---- Kiri: Form input & prediksi ----
with left:
    st.subheader("Input Fitur Mobil")
    with st.form("input_form"):
        form_inputs = {}
        for key, val in example_schema.items():
            if isinstance(val, bool):
                form_inputs[key] = st.checkbox(key, value=val)
            elif isinstance(val, int):
                form_inputs[key] = st.number_input(key, value=int(val))
            elif isinstance(val, float):
                form_inputs[key] = st.number_input(key, value=float(val), format="%f")
            else:
                form_inputs[key] = st.text_input(key, value=str(val))
        submitted = st.form_submit_button("Prediksi Harga")

    if submitted:
        X_df = prepare_input(form_inputs, example_schema)
        try:
            pred = model.predict(X_df)
            price = float(pred[0])
            st.success(f"💰 Prediksi harga mobil: Rp {price:,.2f}")
            st.session_state["last_prediction"] = {
                "price": price,
                "input": X_df.to_dict(orient="records")[0]
            }
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

    if st.checkbox("Tampilkan data input ke model"):
        X_df = prepare_input(form_inputs, example_schema)
        st.dataframe(X_df)

# ---- Kanan: Chat dengan Gemini ----
with right:
    st.subheader("💬 Chat dengan Gemini AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Gemini:** {text}")

    user_input = st.text_area("Ketik pertanyaan kamu di sini:", height=120)
    col1, col2 = st.columns([1, 1])
    with col1:
        ask_btn = st.button("Kirim ke Gemini")
    with col2:
        clear_btn = st.button("Hapus Chat")

    if clear_btn:
        st.session_state.chat_history = []
        st.experimental_rerun()

    if ask_btn and user_input.strip():
        st.session_state.chat_history.append(("user", user_input))

        if not gemini_key_env:
            st.error("❌ Masukkan Gemini API Key di sidebar.")
        else:
            system_prompt = (
                "Kamu adalah asisten AI yang menjelaskan hasil prediksi harga mobil "
                "dan membantu pengguna memahami fitur, model, atau pertanyaan teknis."
            )

            last_pred = st.session_state.get("last_prediction", None)
            if last_pred:
                summary = (
                    f"Terakhir model memprediksi harga: Rp {last_pred['price']:,.2f} "
                    f"untuk mobil dengan fitur: {json.dumps(last_pred['input'], ensure_ascii=False)}."
                )
                system_prompt += " " + summary

            messages = []
            for role, txt in st.session_state.chat_history[-10:]:
                messages.append({
                    "role": "user" if role == "user" else "assistant",
                    "content": txt
                })

            ai_reply = gemini_chat_reply(system_prompt, messages, gemini_key_env)
            st.session_state.chat_history.append(("assistant", ai_reply))
            st.experimental_rerun()

# ---- Footer ----
st.markdown("---")
st.markdown()