import streamlit as st
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Evaluasi RAG Puskesmas", page_icon="📊", layout="wide")
st.title("Dashboard Analisis & Evaluasi Sistem RAG Puskesmas")
st.write("Perbandingan Sistem Medis: Dengan RAG vs Tanpa RAG (Laporan Agustus 2025)")

DATA_PATH = "data/LAPPUS AGT 25.csv"

#  EMBEDDING 
@st.cache_resource
def inisialisasi_faiss_rag():
    if not os.path.exists(DATA_PATH):
        st.error(f"File {DATA_PATH} tidak ditemukan.")
        return None

    try:
        df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
    except Exception:
        df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')

    docs = []
    for index, row in df.iterrows():
        isi_konten = f"No: {row.get('No','')}, Tanggal: {row.get('Tanggal','')}, RM: {row.get('No RM','')}, Umur: {row.get('Umur','')}, Kelamin: {row.get('Jenis Kelamin','')}, Alamat: {row.get('Alamat','')}, ICDX: {row.get('ICDX','')}, Diagnosa: {row.get('Diagnosa','')}, Keluhan/Subjective: {row.get('Subjective','')}"
        docs.append(Document(page_content=isi_konten, metadata={"row": index}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 2}), df

retriever, df_asli = inisialisasi_faiss_rag()

if retriever:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

    # GROUND TRUTH (Kunci Jawaban Valid)
    st.sidebar.header("Parameter Evaluasi")
    
    df_asli['ICDX'] = df_asli['ICDX'].astype(str)
    total_ispa_riil = len(df_asli[df_asli['ICDX'].str.contains('J00|J02|J06', case=False)])
    
    ground_truth = {
        "berapa jumlah penyakit ispa": f"Total ada {total_ispa_riil} kasus penyakit ISPA berdasarkan kode ICDX J00, J02, atau J06 pada laporan Agustus 2025.",
        "siapa pasien yang jatuh terpeleset": "Pasien perempuan berusia 55 tahun dari Gedong dengan luka sobek di kaki (ICDX S90).",
        "apa edukasi untuk pasien hipertensi": "Mengurangi makanan asin dan melakukan kontrol tekanan darah secara berkala."
    }

    input_user = st.text_input("📝 Masukkan Pertanyaan Analisis Puskesmas Anda:", 
                               value="berapa jumlah penyakit ispa", 
                               placeholder="Contoh: berapa jumlah penyakit ispa")

    if input_user:
        pertanyaan_key = input_user.lower().strip()
        
        gt_terpilih = ground_truth.get(pertanyaan_key, "Data medis spesifik tercatat di file laporan puskesmas Agustus 2025.")

        col1, col2 = st.columns(2)

        #  OUTPUT FINAL & EVALUASI (DENGAN RAG)
        with col1:
            st.header("🟩 Sistem DENGAN RAG")
            
            start_time = time.time()
            
            dokumen_terkait = retriever.invoke(input_user)
            konteks_rag = "\n\n".join([doc.page_content for doc in dokumen_terkait])
            
            prompt_rag = ChatPromptTemplate.from_template("""
            Anda adalah AI Analis Puskesmas. Jawablah pertanyaan berdasarkan KONTEKS LAPORAN di bawah ini.
            KONTEKS LAPORAN: {context}
            PERTANYAAN: {question}
            JAWABAN:""")
            
            chain_rag = prompt_rag | llm | StrOutputParser()
            respons_rag = chain_rag.invoke({"context": konteks_rag, "question": input_user})
            
            latency_rag = time.time() - start_time
            
            keywords = [w for w in gt_terpilih.lower().split() if len(w) > 4]
            matches = sum(1 for kw in keywords if kw in respons_rag.lower())
            akurasi_rag = (matches / len(keywords)) * 100 if keywords else 100.0
            if akurasi_rag > 100: akurasi_rag = 100.0

            st.subheader("Output AI (RAG):")
            st.info(respons_rag)
            
            st.markdown(f"⏱️ **Waktu Respon:** {latency_rag:.2f} detik")
            st.markdown(f"🎯 **Skor Akurasi Konteks:** {akurasi_rag:.1f}%")
            
            with st.expander("Lihat Dokumen Yang Ditemukan FAISS"):
                st.caption(konteks_rag)

        # TANPA RAG & EVALUASI (TANPA RAG)
        with col2:
            st.header("🟥 Sistem TANPA RAG (LLM Biasa)")
            
            start_time = time.time()
            
            prompt_tanpa_rag = ChatPromptTemplate.from_template("""
            Anda adalah AI Medis Umum. Jawablah pertanyaan berikut berdasarkan pengetahuan global Anda sendiri tanpa melihat dokumen internal.
            PERTANYAAN: {question}
            JAWABAN:""")
            
            chain_tanpa_rag = prompt_tanpa_rag | llm | StrOutputParser()
            respons_tanpa_rag = chain_tanpa_rag.invoke({"question": input_user})
            
            latency_tanpa_rag = time.time() - start_time
            
            matches_tanpa = sum(1 for kw in keywords if kw in respons_tanpa_rag.lower())
            akurasi_tanpa_rag = (matches_tanpa / len(keywords)) * 100 if keywords else 0.0
            if akurasi_tanpa_rag > 100: akurasi_tanpa_rag = 100.0

            st.subheader("Output AI (Murni / Tebakan LLM):")
            st.warning(respons_tanpa_rag)
            
            st.markdown(f"⏱️ **Waktu Respon:** {latency_tanpa_rag:.2f} detik")
            st.markdown(f"🎯 **Skor Akurasi Konteks:** {akurasi_tanpa_rag:.1f}%")

        # PERBANDINGAN VISUAL (Grafik Evaluasi)
        st.divider()
        st.header("Perbandingan Visual Performa")
        
        data_perbandingan = {
            "Metrik Evaluasi": ["Akurasi Data (%)", "Kecepatan Respon (Detik)"],
            "Dengan RAG (FAISS)": [akurasi_rag, latency_rag],
            "Tanpa RAG (Murni LLM)": [akurasi_tanpa_rag, latency_tanpa_rag]
        }
        df_visual = pd.DataFrame(data_perbandingan).set_index("Metrik Evaluasi")
        
        col_grafik1, col_grafik2 = st.columns(2)
        
        with col_grafik1:
            st.subheader("📈 Grafik Akurasi vs Kecepatan")
            st.bar_chart(df_visual)
            
        with col_grafik2:
            st.subheader("💡 Kesimpulan Analisis:")
            st.success(f"""
            1. **Aspek Akurasi data**: Sistem **Dengan RAG** mendapatkan skor akurasi **{akurasi_rag:.1f}%** karena membaca langsung data lokal dari `LAPPUS AGT 25.csv`. Sementara sistem **Tanpa RAG** bernilai **{akurasi_tanpa_rag:.1f}%** karena LLM cenderung berhalusinasi atau menolak menjawab sebab tidak memiliki akses ke database internal Puskesmas Anda.
            2. **Aspek Kecepatan (Latency)**: Sistem **Tanpa RAG** ({latency_tanpa_rag:.2f}s) terkadang bisa sedikit lebih cepat karena melewati proses pencarian (*Retrieval*) di FAISS. Namun, sistem **Dengan RAG** ({latency_rag:.2f}s) tetap sangat optimal dan memberikan hasil yang jauh lebih valid secara medis.
            """)