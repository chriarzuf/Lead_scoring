import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione della pagina
st.set_page_config(
    page_title="WeRoad Lead Scoring AI",
    page_icon="✈️",
    layout="wide"
)

# --- 1. GENERAZIONE DEI DATI SINTETICI ---
@st.cache_data
def generate_weroad_data(n_samples=1000):
    np.random.seed(42)
    
    # Variabili indipendenti (Features)
    # 1. Minuti passati sul sito (distribuzione normale, media 8 min, deviazione 5)
    time_on_site = np.random.normal(8, 5, n_samples)
    time_on_site = np.clip(time_on_site, 0, 60) # Nessuno sta meno di 0 o più di 60 min per questa demo
    
    # 2. Click sulle email di marketing (distribuzione di Poisson)
    email_clicks = np.random.poisson(lam=1.5, size=n_samples)
    
    # 3. Aggiunta al carrello (Binaria: 0 = No, 1 = Sì)
    # La probabilità aumenta leggermente se passano più tempo sul sito
    prob_cart = 1 / (1 + np.exp(-(time_on_site - 10) / 5)) 
    add_to_cart = np.random.binomial(1, prob_cart)
    
    # Variabile Dipendente (Target): Profittabilità / Lead Score (€)
    # Formula sottostante: Base + (Tempo * peso) + (Email * peso) + (Carrello * peso) + Rumore
    # Ipotizziamo:
    # - Base: 20€
    # - Ogni minuto vale 5€
    # - Ogni click email vale 15€
    # - Aggiungere al carrello vale 300€ (segnale molto forte)
    noise = np.random.normal(0, 30, n_samples) # Un po' di casualità
    profitability = 20 + (5 * time_on_site) + (15 * email_clicks) + (300 * add_to_cart) + noise
    
    data = pd.DataFrame({
        'Minuti_sul_Sito': time_on_site,
        'Click_Email': email_clicks,
        'Aggiunta_Carrello': add_to_cart,
        'Profittabilità_Prevista_€': profitability
    })
    
    return data

# --- INTERFACCIA UTENTE ---

st.title("✈️ WeRoad Lead Scoring: Modello di Regressione Lineare")
st.markdown("""
Questa applicazione simula un modello di Machine Learning per calcolare il valore potenziale (Lead Score) 
di un utente interessato ai viaggi WeRoad, basandosi sul suo comportamento digitale.
""")

# Caricamento dati
data = generate_weroad_data()
X = data[['Minuti_sul_Sito', 'Click_Email', 'Aggiunta_Carrello']]
y = data['Profittabilità_Prevista_€']

# Split e Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metriche
r2 = r2_score(y_test, y_pred)

# --- COLONNE PRINCIPALI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🛠️ Simulatore Utente")
    st.markdown("Imposta i parametri dell'utente per predire il valore.")
    
    input_time = st.slider("Minuti spesi sul sito", 0, 60, 10)
    input_email = st.slider("Click su Email Marketing", 0, 10, 2)
    input_cart = st.radio("Ha aggiunto un viaggio al carrello?", ["No", "Sì"])
    input_cart_val = 1 if input_cart == "Sì" else 0
    
    # Predizione in tempo reale
    input_data = pd.DataFrame([[input_time, input_email, input_cart_val]], 
                              columns=['Minuti_sul_Sito', 'Click_Email', 'Aggiunta_Carrello'])
    prediction = model.predict(input_data)[0]
    
    st.divider()
    st.markdown("### 🎯 Lead Score (Profittabilità)")
    st.metric(label="Valore Stimato Utente", value=f"€ {prediction:.2f}")
    
    if prediction > 400:
        st.success("🔥 Questo è un **Hot Lead**! Contattare subito.")
    elif prediction > 150:
        st.warning("⚠️ Utente interessato. Inviare coupon sconto.")
    else:
        st.info("🧊 Utente freddo. Inserire in campagna nurturing.")

with col2:
    st.subheader("📊 Analisi del Modello")
    
    tab1, tab2, tab3 = st.tabs(["Interpretazione Pesi", "Visualizzazione Dati", "Dataset"])
    
    with tab1:
        st.markdown("Ecco come il modello 'pesa' ogni azione per calcolare il punteggio finale:")
        
        coef_df = pd.DataFrame({
            'Azione (Feature)': ['Ogni minuto sul sito', 'Ogni click email', 'Aggiunta al Carrello'],
            'Impatto sul valore (€)': model.coef_
        })
        
        st.dataframe(coef_df.style.format({'Impatto sul valore (€)': '{:.2f} €'}), use_container_width=True)
        st.markdown(f"""
        **Spiegazione:**
        * L'intercetta (valore base) è **€ {model.intercept_:.2f}**.
        * Il modello spiega il **{r2*100:.1f}%** della varianza nei dati ($R^2$).
        """)
        
    with tab2:
        st.markdown("Relazione tra **Tempo sul Sito** e **Profittabilità**, colorato per chi ha aggiunto al carrello.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=data, 
            x='Minuti_sul_Sito', 
            y='Profittabilità_Prevista_€', 
            hue='Aggiunta_Carrello',
            palette={0: 'grey', 1: '#FF4B4B'}, # Rosso Streamlit per carrello
            alpha=0.6,
            ax=ax
        )
        
        # Disegna la linea di regressione approssimativa per visualizzazione
        m, b = np.polyfit(data['Minuti_sul_Sito'], data['Profittabilità_Prevista_€'], 1)
        plt.plot(data['Minuti_sul_Sito'], m*data['Minuti_sul_Sito'] + b, color='blue', linestyle='--', alpha=0.5, label='Trend Generale')
        
        plt.title("Impatto del Tempo e del Carrello sul Valore")
        plt.xlabel("Minuti sul Sito")
        plt.ylabel("Profittabilità (€)")
        plt.legend(title="Carrello (0=No, 1=Sì)")
        st.pyplot(fig)
        
    with tab3:
        st.markdown("Un'anteprima dei dati generati sinteticamente:")
        st.dataframe(data.head(10), use_container_width=True)

# --- FOOTER ---
st.divider()
st.caption("Demo sviluppata per WeRoad Analytics Case Study.")
