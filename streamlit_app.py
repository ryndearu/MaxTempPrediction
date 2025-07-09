import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import io
import re
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

def read_csv_robust(uploaded_file):
    """Fungsi untuk membaca CSV dengan berbagai strategi parsing"""
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Strategi 1: Baca sebagai text dan perbaiki format decimal
        content = uploaded_file.read().decode('utf-8')
        
        # Perbaiki format decimal yang dikutip: "XX,Y" -> XX.Y
        pattern = r'"(\d+),(\d+)"'
        content = re.sub(pattern, r'\1.\2', content)
        
        # Buat file-like object dari string yang sudah diperbaiki
        fixed_csv = io.StringIO(content)
        df = pd.read_csv(fixed_csv, na_values=['-', '', 'NaN', 'nan', 'null'])
        
        st.success("✅ CSV berhasil dibaca dengan perbaikan format decimal")
        return df
        
    except Exception as e1:
        try:
            # Strategi 2: Coba dengan decimal separator koma
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, decimal=',', na_values=['-', '', 'NaN', 'nan'])
            st.success("✅ CSV berhasil dibaca dengan decimal=','")
            return df
            
        except Exception as e2:
            try:
                # Strategi 3: Format standar dengan titik sebagai decimal
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, decimal='.', na_values=['-', '', 'NaN', 'nan'])
                st.success("✅ CSV berhasil dibaca dengan format standar")
                return df
                
            except Exception as e3:
                # Tampilkan semua error
                st.error("❌ Gagal membaca CSV dengan semua strategi:")
                st.error(f"Strategi 1 (perbaikan format): {str(e1)}")
                st.error(f"Strategi 2 (decimal=','): {str(e2)}")
                st.error(f"Strategi 3 (decimal='.'): {str(e3)}")
                return None

def process_weather_data(df):
    """Memproses data cuaca sesuai dengan preprocessing di notebook"""
    df_processed = df.copy()
    
    # Konversi kolom TANGGAL ke datetime
    try:
        if df_processed['TANGGAL'].dtype == 'object':
            # Coba berbagai format tanggal
            try:
                df_processed['TANGGAL'] = pd.to_datetime(df_processed['TANGGAL'], format='%d-%m-%Y')
            except:
                try:
                    df_processed['TANGGAL'] = pd.to_datetime(df_processed['TANGGAL'], format='%Y-%m-%d')
                except:
                    df_processed['TANGGAL'] = pd.to_datetime(df_processed['TANGGAL'])
        else:
            df_processed['TANGGAL'] = pd.to_datetime(df_processed['TANGGAL'])
    except Exception as e:
        st.error(f"Format tanggal tidak dapat diproses: {str(e)}")
        return None
    
    # Set TANGGAL sebagai index
    df_processed.set_index('TANGGAL', inplace=True)
    
    # Drop kolom yang tidak diperlukan
    if 'DDD_CAR' in df_processed.columns:
        df_processed.drop('DDD_CAR', axis=1, inplace=True)
    
    # Deteksi apakah data sudah memiliki lag features
    has_lag_features = any('_lag' in col for col in df_processed.columns)
    
    if has_lag_features:
        st.info("✅ Data dengan lag features terdeteksi - menggunakan format preprocessing lengkap")
        # Data sudah memiliki lag features, hanya perlu interpolasi
        for col in df_processed.select_dtypes(include=np.number).columns:
            df_processed[col] = df_processed[col].interpolate(method='linear', limit_direction='both')
        
        # Pastikan kolom Tx_target ada
        if 'Tx_target' not in df_processed.columns:
            df_processed['Tx_target'] = df_processed['TX'].shift(-1)
        
        # Pastikan fitur temporal ada
        if 'dayofweek' not in df_processed.columns:
            df_processed['dayofweek'] = df_processed.index.dayofweek
        if 'dayofyear' not in df_processed.columns:
            df_processed['dayofyear'] = df_processed.index.dayofyear
        if 'month' not in df_processed.columns:
            df_processed['month'] = df_processed.index.month
        if 'year' not in df_processed.columns:
            df_processed['year'] = df_processed.index.year
            
    else:
        st.info("📊 Data tanpa lag features terdeteksi - membuat preprocessing otomatis")
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
        help="Upload file CSV dengan format BMKG (contoh: BMKG Jateng - Sheet1.csv)"    )
    
    # Upload file model
    model_file = st.file_uploader(
        "Upload File Model (.joblib)",
        type=['joblib'],
        help="Upload file model Random Forest yang sudah dilatih"
    )
    
    st.markdown('<h2 class="sub-header">📅 Parameter Prediksi</h2>', unsafe_allow_html=True)
    
    # Mode prediksi
    prediction_mode = st.radio(
        "Mode Prediksi:",
        ["🔮 Prediksi dari tanggal tertentu", "📈 Lanjutan dari data terakhir"],
        help="Pilih apakah ingin prediksi dari tanggal spesifik atau melanjutkan dari data terakhir"
    )
    
    if prediction_mode == "🔮 Prediksi dari tanggal tertentu":
        # Input tanggal referensi
        today_date = st.date_input(
            "Tanggal Referensi Prediksi",
            value=datetime.now().date(),
            help="Pilih tanggal sebagai titik awal prediksi. Aplikasi akan mencari data historis terdekat sebelum tanggal ini."
        )
        
        # Option untuk menampilkan data historis
        show_historical_days = st.slider(
            "Tampilkan Data Historis (hari sebelumnya)",
            min_value=5,
            max_value=30,
            value=15,
            help="Berapa hari sebelum tanggal referensi yang akan ditampilkan di grafik"
        )
    else:
        today_date = datetime.now().date()
        show_historical_days = 15
        st.info("📊 Mode lanjutan: Prediksi akan dimulai dari data historis terakhir yang tersedia")    # Input jumlah hari prediksi
    days_to_predict = st.slider(
        "Jumlah Hari Prediksi",
        min_value=1,
        max_value=7,
        value=7,
        help="Pilih berapa hari ke depan yang akan diprediksi (maksimal 7 hari)"
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if csv_file is not None and model_file is not None:
        try:
            # Load dan proses data CSV
            with st.spinner("Memproses data cuaca..."):
                # Baca CSV dengan fungsi robust
                df_raw = read_csv_robust(csv_file)
                
                if df_raw is None:
                    st.error("❌ Gagal membaca file CSV")
                    st.stop()
                
                # Debug: tampilkan info dataset
                st.write(f"📊 Dataset shape: {df_raw.shape}")
                st.write(f"📋 Kolom: {list(df_raw.columns)}")
                
                # Tampilkan preview data
                with st.expander("🔍 Preview Data (5 baris pertama)"):
                    st.dataframe(df_raw.head())
                
                # Proses data
                df_processed = process_weather_data(df_raw)
                
                if df_processed is not None:
                    st.success(f"✅ Data berhasil diproses: {len(df_processed)} baris data")
                      # Debug: tampilkan kolom setelah preprocessing
                    st.write(f"📋 Kolom setelah preprocessing: {list(df_processed.columns)}")
                    
                    # Load model
                    model = joblib.load(model_file)
                    st.success("✅ Model berhasil dimuat")
                    
                    # Ambil fitur model dari data training terakhir
                    # Exclude target variable dari features
                    model_features = [col for col in df_processed.columns if col != 'Tx_target']
                    
                    # Debug: tampilkan fitur model
                    st.write(f"🔧 Features untuk model: {len(model_features)} features")
                    
                    # Tentukan data referensi berdasarkan mode
                    if prediction_mode == "🔮 Prediksi dari tanggal tertentu":
                        # Cari data historis yang sesuai dengan tanggal referensi
                        reference_date = pd.Timestamp(today_date)
                        available_dates = df_processed.index
                        suitable_dates = available_dates[available_dates <= reference_date]
                        
                        if len(suitable_dates) == 0:
                            st.warning(f"⚠️ Tidak ada data historis sebelum tanggal {today_date}. Menggunakan data paling awal.")
                            reference_actual_date = available_dates[0]
                            reference_data = df_processed.iloc[0].to_dict()
                        else:
                            reference_actual_date = suitable_dates[-1]  # Data terdekat sebelum/pada tanggal referensi
                            reference_data = df_processed.loc[reference_actual_date].to_dict()
                        
                        st.info(f"📅 Menggunakan data referensi dari: {reference_actual_date.strftime('%Y-%m-%d')}")
                        mode_info = "dari tanggal referensi yang dipilih"
                        
                    else:  # Mode lanjutan dari data terakhir
                        reference_actual_date = df_processed.index[-1]
                        reference_data = df_processed.iloc[-1].to_dict()
                        # Update today_date untuk mode lanjutan
                        today_date = (reference_actual_date + timedelta(days=1)).date()
                        reference_date = pd.Timestamp(today_date)
                        
                        st.info(f"📅 Melanjutkan prediksi dari data terakhir: {reference_actual_date.strftime('%Y-%m-%d')}")
                        mode_info = "melanjutkan dari data historis terakhir"
                    
                    # Tampilkan informasi data referensi
                    with st.expander("🔍 Data Referensi yang Digunakan"):
                        ref_info = {
                            'Mode Prediksi': mode_info,
                            'Tanggal Referensi': reference_actual_date.strftime('%Y-%m-%d'),
                            'TX (Suhu Maksimum)': f"{reference_data.get('TX', 'N/A'):.1f}°C",
                            'TN (Suhu Minimum)': f"{reference_data.get('TN', 'N/A'):.1f}°C",
                            'TAVG (Suhu Rata-rata)': f"{reference_data.get('TAVG', 'N/A'):.1f}°C",
                            'RH_AVG (Kelembaban)': f"{reference_data.get('RH_AVG', 'N/A'):.1f}%",
                            'RR (Curah Hujan)': f"{reference_data.get('RR', 'N/A'):.1f}mm"
                        }
                        st.json(ref_info)
                    
                    st.markdown('<h2 class="sub-header">📊 Hasil Prediksi</h2>', unsafe_allow_html=True)
                      # Lakukan prediksi iteratif
                    predictions = []
                    current_date = datetime.combine(today_date, datetime.min.time())
                    last_values = reference_data.copy()
                    
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
                    
                    # Tentukan range data historis berdasarkan mode dan tanggal referensi
                    start_pred_date = datetime.strptime(predictions[0]['Tanggal'], '%Y-%m-%d')
                    start_historical_date = reference_date - timedelta(days=show_historical_days)
                    
                    # Filter data historis berdasarkan range tanggal
                    historical_mask = (df_processed.index >= start_historical_date) & (df_processed.index <= reference_date)
                    hist_data = df_processed[historical_mask]
                    
                    if len(hist_data) > 0:
                        fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=hist_data['TX'],
                            mode='lines+markers',
                            name=f'Data Historis ({len(hist_data)} hari)',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6)
                        ))
                        
                        # Tambah marker khusus untuk tanggal referensi
                        if reference_actual_date in hist_data.index:
                            fig.add_trace(go.Scatter(
                                x=[reference_actual_date],
                                y=[hist_data.loc[reference_actual_date, 'TX']],
                                mode='markers',
                                name='Tanggal Referensi',
                                marker=dict(size=12, color='green', symbol='star')                            ))
                    else:
                        st.warning(f"⚠️ Tidak ada data historis dalam rentang {show_historical_days} hari sebelum {today_date}")
                      # Tambah prediksi
                    pred_dates = [datetime.strptime(d['Tanggal'], '%Y-%m-%d') for d in predictions]
                    pred_values = [d['Prediksi_TX'] for d in predictions]
                    
                    # Cek apakah ada data aktual pada tanggal-tanggal prediksi untuk perbandingan
                    actual_on_pred_dates = []
                    actual_values_on_pred_dates = []
                    
                    for pred_date in pred_dates:
                        pred_date_only = pred_date.date()
                        if pred_date_only in df_processed.index:
                            actual_on_pred_dates.append(pred_date)
                            actual_values_on_pred_dates.append(df_processed.loc[pred_date_only, 'TX'])
                    
                    # Tambah garis prediksi
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=pred_values,
                        mode='lines+markers',
                        name=f'Prediksi ({len(predictions)} hari)',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    
                    # Jika ada data aktual pada tanggal prediksi, tambahkan sebagai perpanjangan garis biru
                    if len(actual_on_pred_dates) > 0:
                        fig.add_trace(go.Scatter(
                            x=actual_on_pred_dates,
                            y=actual_values_on_pred_dates,
                            mode='lines+markers',
                            name=f'Data Aktual pada Tanggal Prediksi ({len(actual_on_pred_dates)} titik)',
                            line=dict(color='blue', width=2),
                            marker=dict(size=8, symbol='circle'),
                            connectgaps=False
                        ))
                        
                        # Jika ada data historis sebelumnya, buat garis penghubung ke data aktual pertama pada periode prediksi
                        if len(hist_data) > 0 and reference_actual_date in hist_data.index:
                            connection_to_actual_x = [reference_actual_date, actual_on_pred_dates[0]]
                            connection_to_actual_y = [hist_data.loc[reference_actual_date, 'TX'], actual_values_on_pred_dates[0]]
                            
                            fig.add_trace(go.Scatter(
                                x=connection_to_actual_x,
                                y=connection_to_actual_y,
                                mode='lines',
                                name='Transisi ke Data Aktual',
                                line=dict(color='blue', width=1, dash='dot'),
                                showlegend=False
                            ))                    
                    # Jika ada data historis tetapi tidak ada data aktual pada periode prediksi, buat garis penghubung ke prediksi
                    elif len(hist_data) > 0 and reference_actual_date in hist_data.index:
                        # Garis penghubung dari data referensi ke prediksi pertama
                        connection_x = [reference_actual_date, pred_dates[0]]
                        connection_y = [hist_data.loc[reference_actual_date, 'TX'], pred_values[0]]
                        
                        fig.add_trace(go.Scatter(
                            x=connection_x,
                            y=connection_y,
                            mode='lines',
                            name='Transisi ke Prediksi',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title=f'Prediksi Suhu Maksimum (TX) - {mode_info.title()}<br>Referensi: {reference_actual_date.strftime("%Y-%m-%d")} | <sub>Biru: Historis | Merah: Prediksi | Hijau: Referensi</sub>',
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
                            value=f"{min_pred:.1f}°C"                        )
                    
                    # Analisis perbandingan jika ada data aktual pada periode prediksi
                    if len(actual_on_pred_dates) > 0:
                        st.markdown("### 📊 Analisis Perbandingan Data Aktual vs Prediksi")
                        
                        # Hitung metrics perbandingan
                        comparison_data = []
                        for i, (actual_date, actual_val) in enumerate(zip(actual_on_pred_dates, actual_values_on_pred_dates)):
                            # Cari prediksi untuk tanggal yang sama
                            pred_date_str = actual_date.strftime('%Y-%m-%d')
                            pred_val = None
                            for pred in predictions:
                                if pred['Tanggal'] == pred_date_str:
                                    pred_val = pred['Prediksi_TX']
                                    break
                            
                            if pred_val is not None:
                                difference = actual_val - pred_val
                                comparison_data.append({
                                    'Tanggal': pred_date_str,
                                    'Data Aktual': f"{actual_val:.1f}°C",
                                    'Prediksi': f"{pred_val:.1f}°C",
                                    'Selisih': f"{difference:+.1f}°C",
                                    'Akurasi': f"{100 - abs(difference/actual_val)*100:.1f}%"
                                })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Hitung metrics keseluruhan
                            differences = [float(row['Selisih'].replace('°C', '').replace('+', '')) for row in comparison_data]
                            mae = np.mean([abs(d) for d in differences])
                            rmse = np.sqrt(np.mean([d**2 for d in differences]))
                            
                            col_acc1, col_acc2, col_acc3 = st.columns(3)
                            with col_acc1:
                                st.metric("📈 Mean Absolute Error (MAE)", f"{mae:.2f}°C")
                            with col_acc2:
                                st.metric("📊 Root Mean Square Error (RMSE)", f"{rmse:.2f}°C")
                            with col_acc3:
                                avg_accuracy = np.mean([float(row['Akurasi'].replace('%', '')) for row in comparison_data])
                                st.metric("🎯 Rata-rata Akurasi", f"{avg_accuracy:.1f}%")
                    
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
            
            # Tampilkan informasi debug
            with st.expander("🔍 Debug Information"):
                st.write("Tipe error:", type(e).__name__)
                st.write("Detail error:", str(e))
                if hasattr(e, '__traceback__'):
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        st.markdown('<div class="warning-box">⚠️ Silakan upload file CSV data cuaca dan file model (.joblib) di sidebar untuk memulai prediksi.</div>', unsafe_allow_html=True)
        
        # Tampilkan contoh format data
        st.markdown("### 📋 Format Data CSV yang Diperlukan")
        
        st.markdown("**Format 1: Data Mentah BMKG (akan diproses otomatis)**")
        example_data_simple = {
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
        st.dataframe(pd.DataFrame(example_data_simple), use_container_width=True)
        
        st.markdown("**Format 2: Data dengan Lag Features (preprocessing lengkap)**")
        st.markdown("Data dengan kolom tambahan seperti: `TN_lag1`, `TX_lag1`, `TAVG_lag1`, dll. untuk lag 1-3 hari, serta fitur temporal (`dayofweek`, `dayofyear`, `month`, `year`)")
        
        example_data_processed = {
            'TANGGAL': ['2019-01-04', '2019-01-05'],
            'TN': [25.2, 25.2],
            'TX': [31.0, 31.0], 
            'TAVG': [28.1, 28.5],
            'Tx_target': [31.0, 32.4],
            'TN_lag1': [25.0, 25.2],
            'TX_lag1': [29.8, 31.0],
            'dayofweek': [4, 5],
            'month': [1, 1],
            'year': [2019, 2019],
            '...': ['...', '...']  # Menunjukkan ada lebih banyak kolom
        }
        st.dataframe(pd.DataFrame(example_data_processed), use_container_width=True)

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
