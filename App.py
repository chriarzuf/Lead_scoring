import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione della pagina
st.set_page_config(
    page_title="WeRoad Lead Scoring AI - Advanced",
    page_icon="✈️",
    layout="wide"
)

# --- 1. GENERAZIONE DEI DATI SINTETICI ---
@st.cache_data
def generate_weroad_data(n_samples=1000):
    np.random.seed(42)
    
    # --- FEATURES ESISTENTI ---
    # 1. Minuti passati sul sito (distribuzione normale)
    time_on_site = np.random.normal(8, 5, n_samples)
    time_on_site = np.clip(time_on_site, 0, 60)
    
    # 2. Click sulle email (Poisson)
    email_clicks = np.random.poisson(lam=1.5, size=n_samples)
    
    # 3. Aggiunta al carrello (Binaria 0/1)
    prob_cart = 1 / (1 + np.exp(-(time_on_site - 10) / 5)) 
    add_to_cart = np.random.binomial(1, prob_cart)

    # --- NUOVE FEATURES ---
    # 4. Età (Distribuzione normale centrata sui 29 anni, tipico target WeRoad)
    age = np.random.normal(29, 6, n_samples)
    age = np.clip(age, 18, 55).astype(int)

    # 5. Dispositivo Desktop (1 = Desktop, 0 = Mobile)
    # Diciamo che il 40% usa desktop
    device_desktop = np.random.binomial(1, 0.4, n_samples)

    # 6. Viaggi Passati (0, 1, 2+...)
    # La maggior parte sono nuovi (0), alcuni retention
    past_trips = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.2, 0.08, 0.02])
    
    # --- CALCOLO PROFITTABILITÀ (TARGET) ---
    # Formula: Base + pesi per ogni feature + rumore
    # - Base: 20€
    # - Minuto: +5€
    # - Email: +15€
    # - Carrello: +300€
    # - Età: +3€ per ogni anno (più potere d'acquisto)
    # - Desktop: +50€ (conversione più probabile su ticket alto)
    # - Viaggio Passato: +150€ (alta fedeltà)
    
    noise = np.random.normal(0, 30, n_samples)
    
    profitability = (20 + 
                     (5 * time_on_site) + 
                     (15 * email_clicks) + 
                     (300 * add_to_cart) + 
                     (3 * age) +
                     (50 * device_desktop) +
                     (150 * past_trips) + 
                     noise)
    
    data = pd.DataFrame({
        'Minuti_sul_Sito': time_on_site,
        'Click_Email': email_clicks,
        'Aggiunta_Carrello': add_to_cart,
        'Età': age,
        'Dispositivo_Desktop': device_desktop,
        'Viaggi_Passati': past_trips,
        'Profittabilità_Prevista_€': profitability
    })
    
    return data

# --- INTERFACCIA UTENTE ---

st.title("✈️ WeRoad Lead Scoring AI (V2)")
st.markdown("""
Simulatore avanzato per il calcolo del Lead Score. 
Il modello ora considera anche dati demografici e storici.
""")

# Caricamento dati
data = generate_weroad_data()

# Definizione Features e Target
features = ['Minuti_sul_Sito', 'Click_Email', 'Aggiunta_Carrello', 'Età', 'Dispositivo_Desktop', 'Viaggi_Passati']
X = data[features]
y = data['Profittabilità_Prevista_€']

# Split e Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# --- LAYOUT A COLONNE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🛠️ Profilo Utente")
    st.info("Modifica i parametri per vedere come cambia il valore del lead.")
    
    # Input Utente
    input_time = st.slider("Minuti sul sito", 0, 60, 10)
    input_email = st.slider("Click su Email", 0, 10, 2)
    input_cart = st.toggle("Ha aggiunto al carrello?", value=False)
    
    st.divider()
    
    input_age = st.number_input("Età utente", 18, 60, 29)
    input_device = st.radio("Dispositivo", ["Mobile", "Desktop"], horizontal=True)
    input_past_trips = st.selectbox("Viaggi WeRoad passati", [0, 1, 2, 3])

    # Pre-processing input
    val_cart = 1 if input_cart else 0
    val_device = 1 if input_device == "Desktop" else 0
    
    # Creazione DataFrame input
    input_data = pd.DataFrame([[
        input_time, input_email, val_cart, input_age, val_device, input_past_trips
    ]], columns=features)
    
    # Predizione
    prediction = model.predict(input_data)[0]
    
    st.divider()
    st.markdown("### 🎯 Lead Score Stimato")
    st.metric(label="Valore in Euro", value=f"€ {prediction:.2f}")
    
    # Logica di business
    if prediction > 600:
        st.success("🔥 **SUPER LEAD** (Alta priorità)")
    elif prediction > 350:
        st.warning("⚠️ **Lead Caldo** (Follow-up richiesto)")
    else:
        st.secondary("🧊 **Lead Freddo** (Nurturing)")

with col2:
    st.subheader("📊 Analisi dei Fattori (Coefficienti)")
    
    tab1, tab2 = st.tabs(["Impatto Variabili", "Dataset"])
    
    with tab1:
        st.markdown(f"Accuratezza del modello ($R^2$): **{r2*100:.1f}%**")
        st.write("Questo grafico mostra quanto ogni fattore influenza il prezzo finale (coefficienti della regressione).")
        
        # Creiamo un DF per visualizzare i coefficienti in modo carino
        coef_df = pd.DataFrame({
            'Fattore': features,
            'Peso (€)': model.coef_
        }).sort_values(by='Peso (€)', ascending=False)
        
        # Bar chart orizzontale
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=coef_df, x='Peso (€)', y='Fattore', palette='viridis', ax=ax)
        plt.title("Quanto vale ogni azione/caratteristica?")
        plt.xlabel("Impatto monetario (€)")
        st.pyplot(fig)
        
        st.markdown("""
        **Analisi rapida:**
        * Noterai che **l'Aggiunta al Carrello** e i **Viaggi Passati** sono i predittori più forti.
        * Il **Tempo sul sito** ha un impatto positivo ma minore rispetto alle azioni dirette.
        """)

    with tab2:
        st.write("Campione dei dati generati (prime 10 righe):")
        st.dataframe(data.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.caption("WeRoad Lead Scoring Demo v2.0")
