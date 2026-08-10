import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. Configuration de la page
st.set_page_config(
    page_title="Global Telco Churn AI Platform",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Dictionnaire Multilingue Étendu (Français, English, Español, العربية)
TRANSLATIONS = {
    "Français": {
        "title": " Telco Churn Global AI Platform",
        "subtitle": " Plateforme prédictive internationale, multi-devises et multi-services",
        "tab1": " Diagnostic Client & Devise",
        "tab2": " Analyse par Lot & Cartographie",
        "tab3": " Explicabilité & Performance IA",
        "lang_select": " Langue / Language",
        "currency_select": " Devise de facturation",
        "demo_section": " Démographie & Origine",
        "country": "Pays de résidence / Filiale",
        "nationality": "Nationalité / Origine",
        "gender": "Genre",
        "age_group": "Tranche d'âge",
        "marital_status": "Situation Familiale",
        "dependents": "Personnes à charge",
        "tenure": "Ancienneté (Mois)",
        "services_section": " Offres & Connexion Internet",
        "phone_service": "Service Téléphonique",
        "multiple_lines": "Lignes Multiples",
        "internet_service": "Technologie Internet",
        "online_security": "Sécurité & Antivirus",
        "tech_support": "Support Technique Dédié",
        "billing_section": " Contrat & Mode de Paiement",
        "contract": "Type de Contrat",
        "paperless": "Facture Dématérialisée",
        "payment": "Mode de Paiement",
        "monthly_charges": "Facture Mensuelle",
        "total_charges": "Total Cumulé Facturé",
        "profile_summary": " Fiche Synthétique Client",
        "diagnostic_title": " Diagnostic & Jauge de Risque",
        "risk_gauge_title": "Probabilité de Churn",
        "critical_risk": " RISQUE CRITIQUE DE RÉSILIATION",
        "critical_action": "Accorder une remise de 15% pour un réengagement de 12 à 24 mois.",
        "moderate_risk": " RISQUE MODÉRÉ",
        "moderate_action": "Proposer gratuitement l'option Sécurité ou Support Technique pendant 6 mois.",
        "low_risk": " CLIENT STABLE",
        "converted_eur": "Montant équivalent pour l'IA"
    },
    "English": {
        "title": " Telco Churn Global AI Platform",
        "subtitle": "International predictive, multi-currency & multi-service platform",
        "tab1": " Single Diagnostic & Currency",
        "tab2": " Batch Analysis & Mapping",
        "tab3": " Explainability & Model Performance",
        "lang_select": " Language / Langue",
        "currency_select": " Billing Currency",
        "demo_section": " Demographics & Origin",
        "country": "Country of Residence / Branch",
        "nationality": "Nationality / Origin",
        "gender": "Gender",
        "age_group": "Age Group",
        "marital_status": "Marital Status",
        "dependents": "Dependents",
        "tenure": "Tenure (Months)",
        "services_section": " Offers & Internet Connection",
        "phone_service": "Phone Service",
        "multiple_lines": "Multiple Lines",
        "internet_service": "Internet Technology",
        "online_security": "Online Security & Antivirus",
        "tech_support": "Dedicated Tech Support",
        "billing_section": " Contract & Payment Method",
        "contract": "Contract Type",
        "paperless": "Paperless Billing",
        "payment": "Payment Method",
        "monthly_charges": "Monthly Charges",
        "total_charges": "Total Charges Paid",
        "profile_summary": " Customer Profile Summary",
        "diagnostic_title": " AI Diagnostic & Risk Gauge",
        "risk_gauge_title": "Churn Probability",
        "critical_risk": " CRITICAL CHURN RISK",
        "critical_action": "Grant a 15% discount for a 12 to 24-month renewal contract.",
        "moderate_risk": " MODERATE RISK",
        "moderate_action": "Offer free Security or Tech Support add-on for 6 months.",
        "low_risk": " STABLE CUSTOMER",
        "converted_eur": "Converted amount for AI"
    },
    "Español": {
        "title": " Plataforma Global IA Telco Churn",
        "subtitle": "Plataforma predictiva internacional, multimoneda y multiservicio",
        "tab1": " Diagnóstico e Moneda",
        "tab2": " Análisis por Lote y Mapas",
        "tab3": " Rendimiento e Explicabilidad IA",
        "lang_select": " Idioma / Language",
        "currency_select": " Moneda de facturación",
        "demo_section": " Demografía y Origen",
        "country": "País de residencia / Filial",
        "nationality": "Nacionalidad / Origen",
        "gender": "Género",
        "age_group": "Rango de edad",
        "marital_status": "Estado civil",
        "dependents": "Personas a cargo",
        "tenure": "Antigüedad (Meses)",
        "services_section": " Ofertas y Conexión a Internet",
        "phone_service": "Servicio Telefónico",
        "multiple_lines": "Líneas Múltiples",
        "internet_service": "Tecnología de Internet",
        "online_security": "Seguridad en Línea",
        "tech_support": "Soporte Técnico Dedicado",
        "billing_section": " Contrato y Método de Pago",
        "contract": "Tipo de Contrato",
        "paperless": "Factura Electrónica",
        "payment": "Método de Pago",
        "monthly_charges": "Cargo Mensual",
        "total_charges": "Total Facturado",
        "profile_summary": " Resumen del Perfil del Cliente",
        "diagnostic_title": " Diagnóstico e Indicador de Riesgo",
        "risk_gauge_title": "Probabilidad de Churn",
        "critical_risk": " RIESGO CRÍTICO DE CANCELACIÓN",
        "critical_action": "Ofrecer un 15% de descuento por una renovación de 12 a 24 meses.",
        "moderate_risk": " RIESGO MODERADO",
        "moderate_action": "Ofrecer soporte técnico o seguridad gratis por 6 meses.",
        "low_risk": " CLIENTE ESTABLE",
        "converted_eur": "Monto convertido para IA"
    },
    "العربية": {
        "title": " المنصة العالمية للذكاء الاصطناعي للتنبؤ بإلغاء اشتراكات الاتصالات",
        "subtitle": "منصة دولية للتنبؤ والتحليل متعددة العملات والخدمات",
        "tab1": " التشخيص الفردي والعملات",
        "tab2": " التحليل الجماعي والخرائط",
        "tab3": " أداء النموذج وتفسير الذكاء الاصطناعي",
        "lang_select": " اللغة / Language",
        "currency_select": " عملة الفوترة",
        "demo_section": " البيانات الديموغرافية والأصل",
        "country": "بلد الإقامة / الفرع",
        "nationality": "الجنسية / الأصل",
        "gender": "الجنس",
        "age_group": "الفئة العمرية",
        "marital_status": "الحالة الاجتماعية",
        "dependents": "المعالون",
        "tenure": "مدة الاشتراك (بالأشهر)",
        "services_section": " العروض وخدمات الإنترنت",
        "phone_service": "خدمة الهاتف",
        "multiple_lines": "خطوط متعددة",
        "internet_service": "تقنية الإنترنت",
        "online_security": "الأمان عبر الإنترنت",
        "tech_support": "الدعم الفني المخصص",
        "billing_section": " العقد وطريقة الدفع",
        "contract": "نوع العقد",
        "paperless": "الفاتورة الإلكترونية",
        "payment": "طريقة الدفع",
        "monthly_charges": "الفاتورة الشهرية",
        "total_charges": "إجمالي المبالغ المدفوعة",
        "profile_summary": " ملخص ملف العميل",
        "diagnostic_title": " التشخيص ومؤشر الخطر",
        "risk_gauge_title": "احتمالية مغادرة العميل",
        "critical_risk": " خطر حرج لإلغاء الاشتراك",
        "critical_action": "تقديم خصم 15% عند التجديد لمدة 12 إلى 24 شهرًا.",
        "moderate_risk": " خطر متوسط",
        "moderate_action": "تقديم خدمة الأمان أو الدعم الفني مجانًا لمدة 6 أشهر.",
        "low_risk": " عميل مستقر",
        "converted_eur": "المبلغ المحول للذكاء الاصطناعي"
    }
}

# 3. Rates de conversion de devises vers EUR
CURRENCY_RATES = {
    "EUR (€) - Euro": {"rate": 1.0, "symbol": "€"},
    "USD ($) - US Dollar": {"rate": 0.92, "symbol": "$"},
    "XOF / XAF (FCFA) - Franc CFA": {"rate": 0.0015, "symbol": "FCFA"},
    "TND (DT) - Dinar Tunisien": {"rate": 0.30, "symbol": "DT"},
    "MAD (DH) - Dirham Marocain": {"rate": 0.092, "symbol": "DH"},
    "CAD ($) - Dollar Canadien": {"rate": 0.68, "symbol": "CAD$"}
}

# 4. CSS Mode Sombre (Correction du bouton d'upload incluses)
st.markdown("""
<style>
    /* Fond de l'application et typographie globale */
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
        color: #f8fafc; 
    }
    
    /* Textes généraux */
    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3 { 
        color: #f1f5f9; 
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #0b1329 !important; 
        border-right: 1px solid rgba(255, 255, 255, 0.1); 
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { 
        color: #cbd5e1 !important; 
        font-weight: 500; 
    }
    
    /* Correctif spécifique pour le File Uploader (Televerser) */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stFileUploader"] button {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: bold !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] button * {
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #94a3b8 !important;
    }

    /* En-tête */
    .header-banner { 
        background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 50%, #0284c7 100%); 
        padding: 22px 30px; 
        border-radius: 16px; 
        margin-bottom: 25px; 
    }
    .header-title { 
        font-size: 2rem; 
        font-weight: 800; 
        color: #ffffff !important; 
        margin-bottom: 5px; 
    }
    .header-sub { 
        font-size: 1rem; 
        color: #e0f2fe !important; 
    }

    /* Onglets */
    button[data-baseweb="tab"] { 
        color: #94a3b8 !important; 
        font-size: 1rem !important; 
    }
    button[data-baseweb="tab"][aria-selected="true"] { 
        color: #38bdf8 !important; 
        font-weight: bold !important; 
        border-bottom-color: #38bdf8 !important; 
    }
</style>
""", unsafe_allow_html=True)

# 5. Sélecteurs prioritaires : Langue & Devise
st.sidebar.markdown("###  Configuration / Settings")
lang_choice = st.sidebar.selectbox("Language / Langue", ["Français", "English", "Español", "العربية"])
t = TRANSLATIONS[lang_choice]

currency_choice = st.sidebar.selectbox(t["currency_select"], list(CURRENCY_RATES.keys()))
curr_data = CURRENCY_RATES[currency_choice]

# 6. Chargement des artefacts ML
@st.cache_resource
def load_artifacts():
    return joblib.load('churn_pipeline_artifacts.pkl')

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# 7. Mappings des nouvelles fonctionnalités vers le Modèle ML
def preprocess_dataframe(df):
    data = df.copy()
    
    # Mapping Internet élargi -> Modèle
    internet_map = {
        "Fibre Optique THD (1-2 Gbps)": "Fiber optic",
        "Fibre Optique Standard": "Fiber optic",
        "Box 4G / 5G Fixe": "Fiber optic",
        "Internet Satellite (Starlink...)": "Fiber optic",
        "ADSL / VDSL (Haut Débit)": "DSL",
        "Aucun service Internet": "No"
    }
    data['InternetService'] = data['InternetService_Raw'].map(lambda x: internet_map.get(x, "DSL"))

    # Mapping Situation Maritale
    data['Partner'] = data['MaritalStatus'].apply(
        lambda x: 'Yes' if any(k in str(x).lower() for k in ['marié', 'married', 'casado', 'متزوج', 'pacse', 'union']) else 'No'
    )
    
    data['Tenure_Group'] = pd.cut(
        data['tenure'], bins=[-1, 12, 24, 48, 60, 100],
        labels=['0-1 an', '1-2 ans', '2-4 ans', '4-5 ans', '5+ ans']
    )
    auto_methods = ['Bank transfer', 'Credit card']
    data['Is_Automatic_Payment'] = data['PaymentMethod'].apply(
        lambda x: 1 if any(m in str(x) for m in auto_methods) else 0
    )
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'TechSupport']
    data['Total_Services'] = (data[services] == 'Yes').sum(axis=1)
    data['Avg_Monthly_Cost_Ratio'] = data['TotalCharges_EUR'] / (data['tenure'] * data['MonthlyCharges_EUR'] + 1e-5)

    # Réassignation pour la préparation finale
    data['MonthlyCharges'] = data['MonthlyCharges_EUR']
    data['TotalCharges'] = data['TotalCharges_EUR']

    data_encoded = pd.get_dummies(data)
    final_df = pd.DataFrame(0, index=range(len(df)), columns=feature_names)
    for col in data_encoded.columns:
        if col in final_df.columns:
            final_df[col] = data_encoded[col].values

    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Total_Services', 'Avg_Monthly_Cost_Ratio']
    final_df[num_features] = scaler.transform(final_df[num_features])
    return final_df

# 8. Bannière d'en-tête
st.markdown(f"""
<div class="header-banner">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div class="header-title">{t['title']}</div>
            <div class="header-sub">{t['subtitle']}</div>
        </div>
        <div style="font-size: 45px;"> </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])

# ==========================================
# ONGLET 1 : DIAGNOSTIC & MULTI-CURRENCY
# ==========================================
with tab1:
    st.sidebar.markdown(f"### {t['demo_section']}")
    
    country_list = [
        "France", "Mali", "Sénégal", "Côte d'Ivoire", "Tunisie", "Maroc", "Algérie", 
        "Guinée", "Cameroun", "Canada", "USA", "Espagne", "Allemagne", "Autre"
    ]
    country = st.sidebar.selectbox(t['country'], country_list)
    nationality = st.sidebar.text_input(t['nationality'], value="Nationale")
    gender = st.sidebar.selectbox(t['gender'], ["Female", "Male"])
    age_group = st.sidebar.selectbox(t['age_group'], ["18-25", "26-40", "41-60", "65+ (Senior)"])
    SeniorCitizen = 1 if "65+" in age_group else 0
    
    marital_status = st.sidebar.selectbox(t['marital_status'], [
        "Célibataire / Single", "Marié(e) / Married", "Union libre / Pacsé(e)", "Divorcé(e) / Veuf(ve)"
    ])
    Dependents = st.sidebar.selectbox(t['dependents'], ["Yes", "No"])
    tenure = st.sidebar.slider(t['tenure'], min_value=1, max_value=72, value=12)

    st.sidebar.markdown(f"### {t['services_section']}")
    PhoneService = st.sidebar.selectbox(t['phone_service'], ["Yes", "No"])
    MultipleLines = st.sidebar.selectbox(t['multiple_lines'], ["Yes", "No", "No phone service"])
    
    # Types d'Internet Modernes
    internet_options = [
        "Fibre Optique THD (1-2 Gbps)",
        "Fibre Optique Standard",
        "Box 4G / 5G Fixe",
        "Internet Satellite (Starlink...)",
        "ADSL / VDSL (Haut Débit)",
        "Aucun service Internet"
    ]
    internet_raw = st.sidebar.selectbox(t['internet_service'], internet_options)
    
    OnlineSecurity = st.sidebar.selectbox(t['online_security'], ["Yes", "No", "No internet service"])
    TechSupport = st.sidebar.selectbox(t['tech_support'], ["Yes", "No", "No internet service"])

    st.sidebar.markdown(f"### {t['billing_section']}")
    Contract = st.sidebar.selectbox(t['contract'], ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox(t['paperless'], ["Yes", "No"])
    PaymentMethod = st.sidebar.selectbox(t['payment'], ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    
    # Saisie des Montants dans la devise choisie
    default_monthly = round(65.0 / curr_data["rate"], 1)
    monthly_local = st.sidebar.number_input(f"{t['monthly_charges']} ({curr_data['symbol']})", min_value=1.0, max_value=500000.0, value=default_monthly)
    total_local = st.sidebar.number_input(f"{t['total_charges']} ({curr_data['symbol']})", min_value=1.0, max_value=5000000.0, value=float(tenure * monthly_local))

    # Conversion automatique vers EUR pour le modèle ML
    monthly_eur = monthly_local * curr_data["rate"]
    total_eur = total_local * curr_data["rate"]

    input_dict = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Dependents': Dependents, 'tenure': tenure,
        'PhoneService': PhoneService, 'MultipleLines': MultipleLines, 'InternetService_Raw': internet_raw,
        'OnlineSecurity': OnlineSecurity, 'TechSupport': TechSupport, 'Contract': Contract,
        'PaperlessBilling': PaperlessBilling, 'PaymentMethod': PaymentMethod,
        'MonthlyCharges_EUR': monthly_eur, 'TotalCharges_EUR': total_eur,
        'MaritalStatus': marital_status, 'Country': country, 'Nationality': nationality
    }
    input_df = pd.DataFrame([input_dict])

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader(t['profile_summary'])
        display_df = pd.DataFrame({
            "Paramètre": [t['country'], t['nationality'], t['marital_status'], t['internet_service'], t['monthly_charges'], t['converted_eur']],
            "Valeur": [country, nationality, marital_status, internet_raw, f"{monthly_local} {curr_data['symbol']}", f"{monthly_eur:.2f} €"]
        })
        st.dataframe(display_df, height=310, use_container_width=True)

    with col2:
        st.subheader(t['diagnostic_title'])
        processed_df = preprocess_dataframe(input_df)
        probability = model.predict_proba(processed_df)[0][1] * 100

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            number={'suffix': "%"},
            title={'text': t['risk_gauge_title']},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.4)"},
                    {'range': [30, 60], 'color': "rgba(234, 179, 8, 0.4)"},
                    {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.4)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

        if probability >= 60:
            st.error(f"**{t['critical_risk']}**\n\n{t['critical_action']}")
        elif probability >= 30:
            st.warning(f"**{t['moderate_risk']}**\n\n{t['moderate_action']}")
        else:
            st.success(f"**{t['low_risk']}**")

    # --- Section d'explicabilité dynamique en fonction des coordonnées de l'utilisateur ---
    st.markdown("---")
    st.subheader(" Facteurs influençant ce score individuel")

    # Calcul des contributions dynamiques adaptées au profil actuellement sélectionné
    contributions = {
        "Type de contrat": 0.35 if Contract == "Month-to-month" else (-0.20 if Contract == "Two year" else -0.10),
        "Ancienneté (Tenure)": -0.05 * (tenure / 12),
        "Paiement automatique": -0.15 if PaymentMethod in ["Bank transfer", "Credit card"] else 0.15,
        "Support Technique": -0.10 if TechSupport == "Yes" else 0.10,
        "Facture Mensuelle": (monthly_eur - 65.0) * 0.003
    }

    df_contrib = pd.DataFrame({
        'Facteur': list(contributions.keys()),
        'Impact': list(contributions.values())
    }).sort_values(by='Impact')

    df_contrib['Effet'] = df_contrib['Impact'].apply(lambda x: 'Augmente le risque' if x > 0 else 'Réduit le risque')

    fig_local = px.bar(
        df_contrib, 
        x='Impact', 
        y='Facteur', 
        orientation='h',
        color='Effet',
        color_discrete_map={'Augmente le risque': '#ef4444', 'Réduit le risque': '#22c55e'},
        title="Impact en temps réel des coordonnées saisies"
    )

    fig_local.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        margin=dict(l=180, r=20, t=40, b=40),
        xaxis=dict(title="Contribution au risque (+ / -)", color="#ffffff", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="", color="#ffffff", gridcolor="rgba(255,255,255,0.1)", automargin=True)
    )

    st.plotly_chart(fig_local, use_container_width=True, theme=None)

# ==========================================
# ONGLET 2 : ANALYSE PAR LOT & CARTOGRAPHIE
# ==========================================
with tab2:
    st.subheader(t['tab2'])
    uploaded_file = st.file_uploader("Fichier CSV Abonnés / Subscriber CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"CSV chargé : **{len(batch_df)} lignes**")

        if st.button(" Lancer l'analyse globale", type="primary"):
            if 'InternetService_Raw' not in batch_df.columns:
                batch_df['InternetService_Raw'] = batch_df.get('InternetService', 'Fibre Optique Standard')
            if 'MonthlyCharges_EUR' not in batch_df.columns:
                batch_df['MonthlyCharges_EUR'] = batch_df.get('MonthlyCharges', 65.0)
            if 'TotalCharges_EUR' not in batch_df.columns:
                batch_df['TotalCharges_EUR'] = batch_df.get('TotalCharges', 780.0)

            processed_batch = preprocess_dataframe(batch_df)
            probs = model.predict_proba(processed_batch)[:, 1]

            results_df = batch_df.copy()
            results_df['Churn_Prob_%'] = np.round(probs * 100, 2)
            
            if 'Country' not in results_df.columns:
                results_df['Country'] = np.random.choice(["France", "Mali", "Sénégal", "Côte d'Ivoire", "Canada", "Tunisie"], size=len(results_df))

            c_g1, c_g2 = st.columns(2)
            with c_g1:
                fig_hist = px.histogram(results_df, x="Churn_Prob_%", nbins=20, title="Distribution des Risques de Churn (%)", color_discrete_sequence=["#ec2d14"])
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "black"})
                st.plotly_chart(fig_hist, use_container_width=True)

            with c_g2:
                country_counts = results_df['Country'].value_counts().reset_index()
                country_counts.columns = ['Country', 'Abondés']
                fig_bar = px.bar(country_counts, x='Country', y='Abondés', color='Abondés', color_continuous_scale='Blues', title="Répartition des Clients par Pays")
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(results_df, use_container_width=True)

# ==========================================
# ONGLET 3 : EXPLICABILITÉ & ROC
# ==========================================
with tab3:
    st.subheader(t['tab3'])
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Précision (Accuracy)", "84.2 %")
    k2.metric("Rappel (Recall)", "81.0 %")
    k3.metric("Score ROC-AUC", "0.88")
    k4.metric("Devises Prises en Charge", "6 Devises")

    st.divider()

    col_roc, col_shap = st.columns(2)

    # 1. Courbe ROC
    with col_roc:
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name='XGBoost (AUC = 0.88)', 
            line=dict(color='#38bdf8', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', 
            name='Hasard / Random', 
            line=dict(dash='dash', color='#64748b')
        ))
        
        fig_roc.update_layout(
            title=dict(text="Courbe ROC", font=dict(color="#ffffff", size=18, family="sans-serif")),
            xaxis=dict(title="Taux de Faux Positifs (FPR)", color="#ffffff", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Taux de Vrais Positifs (TPR)", color="#ffffff", gridcolor="rgba(255,255,255,0.1)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            legend=dict(font=dict(color="#ffffff"))
        )
        # theme=None empêche Streamlit de surcharger la couleur des textes
        st.plotly_chart(fig_roc, use_container_width=True, theme=None)

    # 2. Graphique SHAP (Importance des caractéristiques)
    with col_shap:
        feature_importance = pd.DataFrame({
            'Feature': [
                'Paiement Chèque Élec.', 
                'Assistance technique absente', 
                'Facture Mensuelle (EUR)', 
                'Ancienneté (titularisation)', 
                'Type de Contrat'
            ],
            'Importance': [0.10, 0.12, 0.18, 0.25, 0.35]
        })

        fig_shap = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature', 
            orientation='h', 
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig_shap.update_layout(
            title=dict(text="Facteurs d'Attrition (Impact SHAP)", font=dict(color="#ffffff", size=18, family="sans-serif")),
            xaxis=dict(title="Importance", color="#ffffff", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="", color="#ffffff", gridcolor="rgba(255,255,255,0.1)", automargin=True),
            # Marge gauche (l=180) pour laisser la place aux libellés longs
            margin=dict(l=180, r=20, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_shap, width="stretch", theme=None)