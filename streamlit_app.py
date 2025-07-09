import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Suhu Semarang",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown('<h1 class="main-header">🌡️ Prediksi Suhu Kota Semarang</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Aplikasi AI untuk memprediksi suhu maksimum (TX) berdasarkan data historis BMKG menggunakan algoritma Random Forest</div>', unsafe_allow_html=True)

# Fungsi untuk memproses data cuaca
def process_weather_data(df):
    """Memproses data cuaca sesuai dengan preprocessing di notebook"""
    df_processed = df.copy()
    
    # Konversi kolom TANGGAL ke datetime
    try:
        df_processed['TANGGAL'] = pd.to_datetime(df_processed['TANGGAL'], format='%d-%m-%Y')
    except:
        try:
            df_processed['TANGGAL'] = pd.to_datetime(df_processed['TANGGAL'])
        except:
            st.error("Format tanggal tidak dapat diproses. Pastikan format DD-MM-YYYY")
            return None
    
    # Set TANGGAL sebagai index
    df_processed.set_index('TANGGAL', inplace=True)
    
    # Drop kolom yang tidak diperlukan
    if 'DDD_CAR' in df_processed.columns:
        df_processed.drop('DDD_CAR', axis=1, inplace=True)
    
    # Interpolasi nilai hilang
    for col in df_processed.select_dtypes(include=np.number).columns:
        df_processed[col] = df_processed[col].interpolate(method='linear', limit_direction='both')
    
    # Feature Engineering - membuat fitur lag
    df_processed['Tx_target'] = df_processed['TX'].shift(-1)
    lags_to_create = [1, 2, 3]
    weather_features_for_lag = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'DDD_X', 'FF_AVG']
    
    for feature in weather_features_for_lag:
        if feature in df_processed.columns:
            for lag in lags_to_create:
                df_processed[f'{feature}_lag{lag}'] = df_processed[feature].shift(lag)
    
    # Tambah fitur waktu
    df_processed['dayofweek'] = df_processed.index.dayofweek
    df_processed['dayofyear'] = df_processed.index.dayofyear
    df_processed['month'] = df_processed.index.month
    df_processed['year'] = df_processed.index.year
    
    # Hapus baris dengan NaN
    df_processed.dropna(inplace=True)
    
    return df_processed

def create_prediction_input(current_date, last_known_data, df_processed, model_features):
    """Membuat input untuk prediksi berdasarkan tanggal dan data terakhir yang diketahui"""
    
    current_timestamp = pd.Timestamp(current_date)
    
    # Buat dictionary fitur input
    input_data_dict = {
        'dayofweek': current_timestamp.dayofweek,
        'dayofyear': current_timestamp.dayofyear,
        'month': current_timestamp.month,
        'year': current_timestamp.year
    }
    
    # Tambahkan fitur cuaca saat ini (jika tersedia)
    weather_features = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'DDD_X', 'FF_AVG']
    for feature in weather_features:
        if feature in last_known_data:
            input_data_dict[feature] = last_known_data[feature]
        elif feature in df_processed.columns:
            input_data_dict[feature] = df_processed[feature].mean()
        else:
            input_data_dict[feature] = 0
    
    # Tambahkan fitur lag
    for feature in weather_features:
        for lag in [1, 2, 3]:
            lag_feature = f'{feature}_lag{lag}'
            if lag_feature in model_features:
                if feature in last_known_data:
                    input_data_dict[lag_feature] = last_known_data[feature]
                elif feature in df_processed.columns:
                    input_data_dict[lag_feature] = df_processed[feature].mean()
                else:
                    input_data_dict[lag_feature] = 0
    
    # Buat DataFrame dengan urutan kolom yang sama dengan model
    input_df = pd.DataFrame([input_data_dict])
    
    # Pastikan semua kolom model ada
    for col in model_features:
        if col not in input_df.columns:
            if col in df_processed.columns:
                input_df[col] = df_processed[col].mean()
            else:
                input_df[col] = 0
    
    # Reorder kolom sesuai dengan model
    input_df = input_df[model_features]
    
    return input_df

# Sidebar untuk input
with st.sidebar:
    st.markdown('<h2 class="sub-header">📁 Upload File</h2>', unsafe_allow_html=True)
    
    # Upload file CSV
    csv_file = st.file_uploader(
        "Upload File CSV Data Cuaca",
        type=['csv'],
        help="Upload file CSV dengan format BMKG (contoh: BMKG Jateng - Sheet1.csv)"
    )
    
    # Upload file model
    model_file = st.file_uploader(
        "Upload File Model (.joblib)",
        type=['joblib'],
        help="Upload file model Random Forest yang sudah dilatih"
    )
    
    st.markdown('<h2 class="sub-header">📅 Parameter Prediksi</h2>', unsafe_allow_html=True)
    
    # Input tanggal hari ini
    today_date = st.date_input(
        "Tanggal Hari Ini",
        value=datetime.now().date(),
        help="Pilih tanggal referensi untuk memulai prediksi"
    )
    
    # Input jumlah hari prediksi
    days_to_predict = st.slider(
        "Jumlah Hari Prediksi",
        min_value=1,
        max_value=7,
        value=3,
        help="Pilih berapa hari ke depan yang akan diprediksi (maksimal 7 hari)"
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if csv_file is not None and model_file is not None:
        try:
            # Load dan proses data CSV
            with st.spinner("Memproses data cuaca..."):
                df_raw = pd.read_csv(csv_file, decimal=',', na_values=['-', ''])
                df_processed = process_weather_data(df_raw)
                
                if df_processed is not None:
                    st.success(f"✅ Data berhasil diproses: {len(df_processed)} baris data")
                    
                    # Load model
                    model = joblib.load(model_file)
                    st.success("✅ Model berhasil dimuat")
                    
                    # Ambil fitur model dari data training terakhir
                    model_features = df_processed.drop('Tx_target', axis=1).columns.tolist()
                    
                    # Ambil data terakhir yang diketahui
                    last_actual_date = df_processed.index[-1]
                    last_known_data = df_processed.iloc[-1].to_dict()
                    
                    st.markdown('<h2 class="sub-header">📊 Hasil Prediksi</h2>', unsafe_allow_html=True)
                    
                    # Lakukan prediksi iteratif
                    predictions = []
                    current_date = datetime.combine(today_date, datetime.min.time())
                    last_values = last_known_data.copy()
                    
                    for i in range(days_to_predict):
                        pred_date = current_date + timedelta(days=i)
                        
                        # Buat input untuk prediksi
                        input_df = create_prediction_input(pred_date, last_values, df_processed, model_features)
                        
                        # Lakukan prediksi
                        predicted_tx = model.predict(input_df)[0]
                        
                        predictions.append({
                            'Tanggal': pred_date.strftime('%Y-%m-%d'),
                            'Hari': pred_date.strftime('%A'),
                            'Prediksi_TX': round(predicted_tx, 2)
                        })
                        
                        # Update nilai terakhir untuk iterasi selanjutnya
                        last_values['TX'] = predicted_tx
                        last_values['TN'] = last_values.get('TN', df_processed['TN'].mean())
                        last_values['TAVG'] = (last_values['TN'] + predicted_tx) / 2
                        last_values['RH_AVG'] = last_values.get('RH_AVG', df_processed['RH_AVG'].mean())
                        last_values['RR'] = 0  # Asumsi tidak ada hujan
                        last_values['SS'] = df_processed['SS'].mean()  # Rata-rata sunshine
                        last_values['FF_X'] = last_values.get('FF_X', df_processed['FF_X'].mean())
                        last_values['DDD_X'] = last_values.get('DDD_X', df_processed['DDD_X'].mean())
                        last_values['FF_AVG'] = last_values.get('FF_AVG', df_processed['FF_AVG'].mean())
                    
                    # Tampilkan hasil dalam tabel
                    results_df = pd.DataFrame(predictions)
                    
                    # Format tabel dengan styling
                    st.markdown("### 📋 Tabel Prediksi")
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualisasi dengan Plotly
                    st.markdown("### 📈 Grafik Prediksi")
                    
                    fig = go.Figure()
                    
                    # Tambah data historis (10 hari terakhir)
                    hist_data = df_processed.tail(10)
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['TX'],
                        mode='lines+markers',
                        name='Data Historis',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Tambah prediksi
                    pred_dates = [datetime.strptime(d['Tanggal'], '%Y-%m-%d') for d in predictions]
                    pred_values = [d['Prediksi_TX'] for d in predictions]
                    
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=pred_values,
                        mode='lines+markers',
                        name='Prediksi',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    
                    fig.update_layout(
                        title='Prediksi Suhu Maksimum (TX) Kota Semarang',
                        xaxis_title='Tanggal',
                        yaxis_title='Suhu (°C)',
                        hovermode='x unified',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Informasi tambahan
                    avg_pred = np.mean(pred_values)
                    max_pred = max(pred_values)
                    min_pred = min(pred_values)
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.metric(
                            label="📊 Rata-rata Prediksi",
                            value=f"{avg_pred:.1f}°C"
                        )
                    
                    with col_info2:
                        st.metric(
                            label="🌡️ Suhu Tertinggi",
                            value=f"{max_pred:.1f}°C"
                        )
                    
                    with col_info3:
                        st.metric(
                            label="❄️ Suhu Terendah",
                            value=f"{min_pred:.1f}°C"
                        )
                    
                    # Download hasil prediksi
                    st.markdown("### 💾 Download Hasil")
                    csv_download = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_download,
                        file_name=f"prediksi_suhu_semarang_{today_date}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error dalam memproses data: {str(e)}")
            st.markdown('<div class="error-box">Pastikan format file CSV sesuai dengan contoh BMKG dan model telah dilatih dengan benar.</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="warning-box">⚠️ Silakan upload file CSV data cuaca dan file model (.joblib) di sidebar untuk memulai prediksi.</div>', unsafe_allow_html=True)
        
        # Tampilkan contoh format data
        st.markdown("### 📋 Format Data CSV yang Diperlukan")
        example_data = {
            'TANGGAL': ['01-01-2019', '02-01-2019', '03-01-2019'],
            'TN': [24.8, 25.6, 25.0],
            'TX': [31.0, 31.0, 29.8],
            'TAVG': [27.8, 27.5, 27.5],
            'RH_AVG': [82.0, 84.0, 85.0],
            'RR': [3.2, 0.0, 1.6],
            'SS': [5.4, 2.5, 2.6],
            'FF_X': [8.0, 5.0, 4.0],
            'DDD_X': [350.0, 270.0, 10.0],
            'FF_AVG': [3.0, 2.0, 1.0]
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)

with col2:
    st.markdown('<h2 class="sub-header">ℹ️ Informasi</h2>', unsafe_allow_html=True)
    
    with st.expander("📖 Tentang Aplikasi"):
        st.markdown("""
        **Prediksi Suhu Semarang** adalah aplikasi AI yang menggunakan algoritma Random Forest 
        untuk memprediksi suhu maksimum (TX) berdasarkan data historis cuaca BMKG.
        
        **Fitur:**
        - Upload data cuaca dalam format CSV
        - Upload model Random Forest yang sudah dilatih
        - Prediksi suhu untuk 1-7 hari ke depan
        - Visualisasi interaktif
        - Download hasil prediksi
        """)
    
    with st.expander("📊 Parameter Data"):
        st.markdown("""
        **Kolom yang diperlukan:**
        - **TANGGAL**: Format DD-MM-YYYY
        - **TN**: Suhu minimum (°C)
        - **TX**: Suhu maksimum (°C)
        - **TAVG**: Suhu rata-rata (°C)
        - **RH_AVG**: Kelembaban rata-rata (%)
        - **RR**: Curah hujan (mm)
        - **SS**: Lama penyinaran matahari (jam)
        - **FF_X**: Kecepatan angin maksimum (m/s)
        - **DDD_X**: Arah angin (derajat)
        - **FF_AVG**: Kecepatan angin rata-rata (m/s)
        """)
    
    with st.expander("🤖 Tentang Model"):
        st.markdown("""
        **Random Forest Regressor** dengan parameter:
        - n_estimators: 100
        - max_depth: None
        - min_samples_split: 2
        - min_samples_leaf: 1
        - random_state: 42
        
        Model menggunakan fitur lag (1-3 hari sebelumnya) dan fitur temporal 
        untuk meningkatkan akurasi prediksi.
        """)
    
    with st.expander("⚠️ Catatan Penting"):
        st.markdown("""
        - Akurasi prediksi menurun untuk hari yang lebih jauh
        - Model memerlukan data historis minimal 3 hari
        - Prediksi diasumsikan tidak ada hujan untuk hari berikutnya
        - Fitur cuaca selain TX menggunakan rata-rata historis
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🌡️ Aplikasi Prediksi Suhu Semarang | Powered by Random Forest & Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)
