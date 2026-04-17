import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="SAGE-ShapFed Dashboard")
st.title("Federated Learning İzleme Paneli")

# Log dosyasını okuma (Eğitim kodunun yazdığı json veya csv)
log_path = "./results/HAM10000/dashboard_data.json" # Kendi yoluna göre ayarla

if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    # En son round bilgisini al
    last_round = max([int(k) for k in data.keys()])
    st.subheader(f"Gözlemlenen Son Round: {last_round}")
    
    round_data = data[str(last_round)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### İstemci Veri Dağılımı (Ground Truth)")
        # Veriyi DataFrame'e çevirip ısı haritası (heatmap) yapıyoruz
        # X ekseni Sınıflar, Y ekseni Client'lar olacak şekilde
        df_clients = pd.DataFrame(round_data['client_distributions']).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_clients, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("### Global Model Tahmin Yönelimi")
        st.caption("Eğer burada sadece 'nv' görüyorsan, Majority Class Dominance sorunu devam ediyor demektir.")
        df_preds = pd.DataFrame(round_data['global_predictions'].items(), columns=['Sınıf', 'Tahmin Sayısı'])
        st.bar_chart(df_preds.set_index('Sınıf'))

    st.markdown("### Pseudo-Label Baskınlığı (Unlabeled Data)")
    st.info("FixMatch mekanizmasının hangi sınıfları 'güvenilir' bulduğunu gösterir.")
    df_pseudo = pd.DataFrame(round_data['pseudo_labels'].items(), columns=['Sınıf', 'Sahte Etiket Sayısı'])
    st.bar_chart(df_pseudo.set_index('Sınıf'))

else:
    st.warning("Henüz log dosyası oluşmadı veya bulunamadı. Eğitimin bir round tamamlamasını bekleyin.")
