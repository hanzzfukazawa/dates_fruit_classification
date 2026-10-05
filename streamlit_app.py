import streamlit as st
import pandas as pd
import joblib

model = joblib.load('dates_fruit_model.joblib')

st.set_page_config(
    page_title="Klasifikasi Kurma",
    page_icon=":palm_tree:"
)

st.title(":palm_tree: Klasifikasi Kurma")
st.markdown("Web aplikasi machinelearning classification untuk memprediksi stage kurma.")
ukuran_cm = st.slider("Ukuran (cm)", 1.62-1.00, 6.11+1.00, 2.89)
berat_g = st.slider("Berat (g)", 3.43-1.00, 20.10+1.00, 10.30)
kekerasan_N = st.slider("Kekerasan (N)", 2.00-1.00, 89.94+1.00, 2.60)
kadar_gula_brix = st.slider("Kadar gula (brix)", 13.80-1.00, 82.81+1.00, 72.15)
kadar_air_pct = st.slider("Kadar air (pct)", 10.05-1.00, 79.99+1.00, 12.45)
varietas = st.selectbox("Varietas", ['Ajwa', 'Barhi', 'Deglet Noor', 'Medjool', 'Zahidi'], 0)
tingkat_warna = st.selectbox("Tingkat warna", ['Cokelat Muda', 'Cokelat Tua', 'Hitam', 'Kuning'], 1)
grade_cacat = st.selectbox("Grade cacat", ['Berat', 'Ringan', 'Sedang', 'Tidak Ada'], 0)
if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame([[ukuran_cm,berat_g,kekerasan_N,kadar_gula_brix,kadar_air_pct,varietas,tingkat_warna,grade_cacat]], columns=['ukuran_cm', 'berat_g', 'kekerasan_N', 'kadar_gula_brix', 'kadar_air_pct', 'varietas', 'tingkat_warna', 'grade_cacat'])
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}**")
    st.balloons()

st.divider()
st.caption("Dibuat dengan :fire: oleh **Hanzz**")