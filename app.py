import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. ตั้งค่าหน้าตาของเว็บ (Layout)
st.set_page_config(page_title="Retail Smart AI", layout="wide")

# 2. โหลดข้อมูล (Caching เพื่อให้โหลดเร็วขึ้น)
@st.cache_resource
def load_all_data():
    P = joblib.load('P_matrix.joblib')
    Q = joblib.load('Q_matrix.joblib')
    product_lookup = joblib.load('product_lookup.joblib')
    R_final = joblib.load('R_final.joblib')
    return P, Q, product_lookup, R_final

P, Q, product_lookup, R_final = load_all_data()

# 3. ส่วนหัวของหน้าเว็บ
st.title("🛒 ระบบแนะนำสินค้าอัจฉริยะ (NMF Algorithm)")
st.markdown("---")

# 4. ส่วนแถบด้านข้าง (Sidebar) สำหรับเลือกข้อมูล
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    # ให้เลือก User ID ตามลำดับ Index 0 ถึงจำนวนลูกค้าทั้งหมด
    user_id = st.number_input("ใส่รหัสลำดับลูกค้า (User Index):", min_value=0, max_value=len(P)-1, value=0)
    top_n = st.slider("ต้องการคำแนะนำกี่รายการ?", 1, 10, 5)
    
    st.info("💡 หมายเหตุ: ระบบจะคำนวณจากคุณลักษณะแฝง (Latent Features) จำนวน 40 กลุ่มที่คุณเทรนมา")

# 5. ปุ่มเริ่มการคำนวณ
if st.button('🚀 เริ่มประมวลผลการแนะนำ'):
    # --- ขั้นตอนการคำนวณ (Recommendation Logic) ---
    # คำนวณหาคะแนน (Dot Product ระหว่างลูกค้าคนนั้นกับสินค้าทั้งหมด)
    user_scores = P[user_id].dot(Q)
    
    # ดึง Index ของสินค้าที่ได้คะแนนสูงสุด Top-N
    recommended_indices = user_scores.argsort()[-top_n:][::-1]
    
    # แบ่งคอลัมน์แสดงผล
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 สินค้าที่ระบบแนะนำสำหรับคุณ")
        for i, idx in enumerate(recommended_indices):
            # ดึงข้อมูลจากไฟล์ product_lookup ที่เราเตรียมไว้
            row = product_lookup.iloc[idx]
            name = row['product_name']
            p_id = row['product_id']
            st.write(f"**{i+1}. {name}**")
            st.caption(f"รหัสสินค้า: {p_id}")
            
    with col2:
        st.subheader("📜 ประวัติการซื้อเดิม (Top History)")
        # ดึงประวัติการซื้อจริงจาก R_final
        user_history = R_final[user_id].toarray().flatten()
        past_indices = np.where(user_history > 0)[0]
        
        # แสดงรายการที่เคยซื้อ (โชว์ไม่เกิน 10 รายการ)
        if len(past_indices) == 0:
            st.write("ไม่มีประวัติการซื้อสำหรับลูกค้ารายนี้")
        else:
            for idx in past_indices[:10]:
                name = product_lookup.iloc[idx]['product_name']
                st.write(f"- {name}")

# 6. ส่วนท้าย (Footer)
st.markdown("---")
st.caption("พัฒนาโดย: โปรเจกต์วิจัยการแนะนำสินค้าด้วยวิธี Matrix Factorization")