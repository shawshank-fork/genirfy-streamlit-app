# music_genre_app.py

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.image import resize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import librosa.display
import requests
from streamlit_lottie import st_lottie

# music_genre_app.py

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.image import resize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import librosa.display
import requests
from streamlit_lottie import st_lottie

# ✅ Set layout and page title
st.set_page_config(page_title="Genirfy - Music Genre Classifier", layout="wide")

# ✅ Load Material Symbols globally
def set_global_material_font():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

        .material-symbols-outlined, .st-emotion-cache-1cypcdb, .st-emotion-cache-1gulkj5 {
            font-family: 'Material Symbols Outlined';
            font-weight: normal;
            font-style: normal;
            font-size: 24px;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            white-space: nowrap;
            direction: ltr;
            -webkit-font-smoothing: antialiased;
            font-variation-settings:
                'FILL' 0,
                'wght' 400,
                'GRAD' 0,
                'opsz' 48;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call it immediately after defining
set_global_material_font()

# ✅ Lottie JSON loader
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Global font styling
def set_custom_font():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bitcount+Prop+Single:wght@100..900&display=swap');
       
        html, body, [class*="css"] {
            font-family: 'Bitcount Prop Single', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6, p, li, span, div {
            font-family: 'Bitcount Prop Single', sans-serif !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Background styling
def set_home_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #0e1117);
            color: white;
        }

        h1, h2, h3, h4, h5, h6, p, li {
            color: white !important;
        }

        .stMarkdown img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 90%;
            max-width: 800px;
            border-radius: 12px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.6);
        }

        .custom-title {
            font-family: 'Bitcount Prop Single', sans-serif;
            font-size: 48px;
            font-weight: 500;
            color: #f7f7f7;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
            letter-spacing: 1px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #111112, #29294d);
            color: white;
        }

        section[data-testid="stSidebar"] .css-1cypcdb,
        section[data-testid="stSidebar"] .css-1cpxqw2,
        section[data-testid="stSidebar"] .css-10trblm {
            color: white !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: #e3dcdc;
        }

        section[data-testid="stSidebar"] label {
            color: #ccc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./Trained_model.h5")

model = load_model()
classes = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
genre_descriptions = {
    'blues': "Slow tempo and expressive melodies.",
    'classical': "Orchestral pieces, symphonies, and structured compositions.",
    'country': "Acoustic guitar, storytelling lyrics, and twangy vocals.",
    'disco': "Funky beats, 70s dance grooves, and vibrant rhythms.",
    'hiphop': "Beats, raps, loops, and urban storytelling.",
    'jazz': "Swing, improvisation, and soulful instruments.",
    'metal': "Loud, distorted guitars, double bass drums, and aggressive vocals.",
    'pop': "Catchy hooks, mainstream appeal, and modern production.",
    'reggae': "Laid-back rhythm, off-beat accents, and Jamaican roots.",
    'rock': "Electric guitars, drums, and high-energy performances."
}

# Preprocessing
def load_and_preprocess_file(file_path, target_shape=(210,210)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    chunk_overlap = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = chunk_overlap * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data), sample_rate, audio_data

# Model prediction
def model_prediction(X_test, top_k=3, threshold=0.15):
    y_pred = model.predict(X_test, verbose=0)
    summed_preds = np.sum(y_pred, axis=0)
    normalized_preds = summed_preds / np.sum(summed_preds)
    top_indices = np.argsort(normalized_preds)[::-1]
    selected = [i for i in top_indices[:top_k] if normalized_preds[i] >= threshold]
    return selected, normalized_preds, y_pred

# Spectrogram
def plot_spectrogram(audio_data, sr):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, ax=ax, cmap='viridis')
    ax.set(title='Mel Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(fig)

# HOME PAGE
def home_page():
    set_global_material_font()
    set_custom_font()
    set_home_background()

    st.image("GENIRFY.gif", use_container_width=True)

    st.markdown("""
**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Prediction** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one or more of the predefined genres.
3. **Results:** View the predicted genres along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Prediction** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!
""")

    st.markdown("### About Us")
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("""
Learn more about the project, our team, and our mission on the **About Project** page.
        """)

    with right_col:
        about_lottie = load_lottieurl("https://lottie.host/1eb5a931-70b9-4c41-a26e-9996fc39562b/WjTe1pKIpj.json")
        if about_lottie:
            st_lottie(about_lottie, height=220, key="about_home_lottie")

# PREDICTION PAGE
def prediction_page():
    set_global_material_font()
    set_custom_font()
    set_home_background()

    st.title("🔍 Predict Music Genre")
    st.write("Upload a WAV or MP3 file to classify its genre:")
    lottie_coding = load_lottieurl("https://lottie.host/e7fea65d-a1aa-40d7-92a0-d22aed79ecea/ppNq2Gfi24.json")
    left_col, right_col = st.columns([2, 1])
    with right_col:
        if lottie_coding:
            st_lottie(lottie_coding, height=250, key="coding_predict")

    uploaded_file = st.file_uploader("Choose a music file", type=["wav", "mp3"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file, format='audio/mp3')
        with st.spinner("🔬 Processing audio and predicting..."):
            X_test, sr, audio_data = load_and_preprocess_file(file_path)
            c_indices, genre_probs, y_pred_all = model_prediction(X_test)

        st.success("✅ Prediction complete!")
        st.balloons()

        st.subheader("🎧 Top Predicted Genre(s):")
        for i in c_indices:
            st.markdown(f"**{classes[i].title()}** ({genre_probs[i]*100:.2f}%)")
            st.caption(genre_descriptions[classes[i]])

        st.subheader("📊 Confidence Chart:")
        df = pd.DataFrame({"Genre": classes, "Probability": genre_probs}).sort_values("Probability", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x="Probability", y="Genre", data=df, palette="coolwarm", ax=ax)
        ax.set_xlim(0, 1)
        st.pyplot(fig)

        st.subheader("🖼️ Spectrogram Preview")
        plot_spectrogram(audio_data, sr)

        st.subheader("🧠 Chunk-wise Predictions")
        chunkwise = np.argmax(y_pred_all, axis=1)
        chunk_counts = pd.Series(chunkwise).value_counts(normalize=True)
        chunk_dist = pd.DataFrame({
            "Genre": [classes[i] for i in chunk_counts.index],
            "Proportion": chunk_counts.values
        })

        st.dataframe(chunk_dist)

        csv = chunk_dist.to_csv(index=False).encode('utf-8')
        st.download_button("📅 Download Prediction Report", data=csv, file_name="genre_prediction_report.csv", mime='text/csv')

        os.remove(file_path)

# ABOUT PAGE
def about_page():
    set_global_material_font()
    set_custom_font()
    set_home_background()

    st.title("ℹ️ About GENIRFY")
    st.markdown("""
Music, Experts have been trying for a long time to understand sound and what differentiates one song from another. How to visualize sound. What makes a tone different from another.

This data hopefully can give the opportunity to do just that.

### About Dataset
#### Content
1. **genres original** - A collection of 10 genres with 100 audio files each  
2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock  
3. **images original** - Audio files converted to **Mel Spectrograms** for CNN-based classification  
4. **2 CSV files** - One per-song, one with 3-sec slices  

---

This AI-powered app uses a **Convolutional Neural Network (CNN)** trained on the **GTZAN dataset** to classify music genres using mel-spectrograms.
""")
    st.markdown("---")
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown("**Key Features:**")
        st.markdown("""
- 🎵 Drag & drop audio upload  
- 🧠 Top-genre predictions  
- 📊 Confidence bar chart  
- 🎼 Mel spectrogram preview  
- 🔄 Chunk-wise prediction breakdown  
- 📀 Downloadable CSV report
        """)
    with right_col:
        animation_json = load_lottieurl("https://lottie.host/09780275-02f5-4807-a912-e8a16524954a/zELeen5fw8.json")
        if animation_json:
            st_lottie(animation_json, height=300, key="about_animation")

    st.markdown("---\nBuilt using TensorFlow, Librosa, and Streamlit.")

# PAGE ROUTING
page = st.sidebar.selectbox("📁 Dashboard", ["Home", "Prediction", "About Project"])

if page == "Home":
    home_page()
elif page == "Prediction":
    prediction_page()
elif page == "About Project":
    about_page()
