import os
from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Fungsi untuk mengubah halaman menjadi tampilan yang lebih menarik


def set_page_layout():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(135deg, #f3ec78 0%, #af4261 100%);
            color: #333333;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #232526 0%, #414345 100%);
            color: #ffffff;
        }
        .sidebar .sidebar-content .stButton>button {
            background-color: #64dfdf;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .sidebar .sidebar-content .stButton>button:hover {
            background-color: #5bb6b6;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk menggambar diagram dan menampilkan tabel


def draw_charts(df):
    car = df[df['jenis_kendaraan'] == 'car']['masuk'].sum()
    truck = df[df['jenis_kendaraan'] == 'truck']['masuk'].sum()
    total = car + truck

    labels = ['Car', 'Truck']
    sizes = [car, truck]
    persentase_cars = car / total * 100 if total > 0 else 0
    persentase_trucks = truck / total * 100 if total > 0 else 0

    data = {
        'Kategori': ['Car', 'Truck'],
        'Jumlah': [car, truck],
        'Persentase': [f'{persentase_cars:.2f}%', f'{persentase_trucks:.2f}%']
    }

    if os.path.isfile('history.csv'):
        history_df = pd.read_csv('history.csv')
        history_df = pd.concat(
            [history_df, pd.DataFrame([data])], ignore_index=True)
    else:
        history_df = pd.DataFrame([data])
    history_df.to_csv('history.csv', index=False)

    fig, ax = plt.subplots()
    explode = (0, 0.1)
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           startangle=90, colors=['#ff9999', '#66b3ff'])
    ax.axis('equal')
    st.sidebar.pyplot(fig)

    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=['#ff9999', '#66b3ff'])
    ax.set_ylabel('Jumlah')
    ax.set_title('Jumlah Kendaraan (Bar Chart)')
    st.sidebar.pyplot(fig)

    fig, ax = plt.subplots()
    ax.fill_between(labels, sizes, color='skyblue', alpha=0.4)
    ax.plot(labels, sizes, marker='o', color='Slateblue', alpha=0.6)
    ax.set_ylabel('Jumlah')
    ax.set_title('Jumlah Kendaraan (Area Chart)')
    st.sidebar.pyplot(fig)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           startangle=90, wedgeprops=dict(width=0.3), colors=['#ff9999', '#66b3ff'])
    ax.axis('equal')
    st.sidebar.pyplot(fig)

    table = pd.DataFrame(data)

    st.write("Total Objek Terdeteksi:")
    st.table(table)

# Fungsi untuk menampilkan data history


def show_history(history_df):
    if history_df.empty:
        st.write("Belum ada data history yang tersimpan.")
    else:
        st.write("### History")
        for index, row in history_df.iterrows():
            if st.button(f"Hapus Baris {index}"):
                history_df.drop(index, inplace=True)
                history_df.to_csv('history.csv', index=False)
                st.success(f"Baris {index} berhasil dihapus dari history.")
            st.write(row)

# Fungsi untuk mengambil data dari MongoDB


def load_data_from_mongodb():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["db_jenis_kendaraan"]
    collection = db["hasil_deteksi"]
    data = list(collection.find())
    for document in data:
        document['_id'] = str(document['_id'])
    df = pd.DataFrame(data)
    return df


# Set tata letak halaman dan warna latar belakang
set_page_layout()

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home', 'Dataframe', 'History')
)

if option == 'Home' or option == '':
    st.title("Selamat Datang di Halaman Utama")
    st.write("Di sini Anda dapat memilih untuk melihat data atau visualisasi.")
elif option == 'Dataframe':
    st.title("Dataframe")
    try:
        df = load_data_from_mongodb()
        st.dataframe(df)
        draw_charts(df)
        if st.button('Simpan Data Visualisasi ke History'):
            st.write("Data visualisasi berhasil disimpan ke history.")
    except Exception as e:
        st.error(f"Gagal mengambil data dari MongoDB: {e}")
elif option == 'History':
    st.title("History")
    try:
        history_df = pd.read_csv('history.csv')
    except FileNotFoundError:
        history_df = pd.DataFrame(columns=['Kategori', 'Jumlah', 'Persentase'])
    show_history(history_df)
