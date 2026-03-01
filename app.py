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
    # โหลดรายชื่อ ID ลูกค้าที่เรา Export มาเพิ่ม
    customer_ids = joblib.load('customer_ids.joblib') 
    return P, Q, product_lookup, R_final, customer_ids

P, Q, product_lookup, R_final, customer_ids = load_all_data()

# 3. ส่วนหัวของหน้าเว็บ
st.title("🛒 ระบบแนะนำสินค้าอัจฉริยะ (NMF Algorithm)")
st.markdown("---")

# 4. ส่วนแถบด้านข้าง (Sidebar) สำหรับเลือกข้อมูล
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    
    # แก้ไข: เปลี่ยนจากใส่ตัวเลข เป็นการเลือกจาก Customer ID จริง
    selected_customer_id = st.selectbox(
        "เลือกรหัสลูกค้า (Customer ID):", 
        options=customer_ids,
        help="คุณสามารถพิมพ์ค้นหารหัสลูกค้าในช่องนี้ได้เลย"
    )
    
    # แปลง ID ที่เลือกกลับเป็นลำดับ (Index) เพื่อใช้คำนวณกับ Matrix P
    user_id = customer_ids.index(selected_customer_id)
    
    top_n = st.slider("ต้องการคำแนะนำกี่รายการ?", 1, 10, 5)
    
    st.divider()
    st.info(f"💡 ลูกค้าลำดับที่: {user_id}\n\nระบบคำนวณจาก Latent Features 40 กลุ่ม")

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
            # ดึงข้อมูลจากไฟล์ product_lookup
            row = product_lookup.iloc[idx]
            name = row['product_name']
            p_id = row['product_id']
            
            with st.container():
                st.write(f"**{i+1}. {name}**")
                st.caption(f"รหัสสินค้า: {p_id}")
            
    with col2:
        st.subheader("📜 ประวัติการซื้อเดิม (Top History)")
        # ดึงประวัติการซื้อจริงจาก R_final
        user_history = R_final[user_id].toarray().flatten()
        past_indices = np.where(user_history > 0)[0]
        
        # แสดงรายการที่เคยซื้อ (โชว์ไม่เกิน 10 รายการ)
        if len(past_indices) == 0:
            st.warning("ไม่มีประวัติการซื้อสำหรับลูกค้ารายนี้")
        else:
            for idx in past_indices[:10]:
                name = product_lookup.iloc[idx]['product_name']
                st.write(f"✅ {name}")

# 6. ส่วนท้าย (Footer)
st.markdown("---")
st.caption(f"สถานะระบบ: ออนไลน์ | จำนวนลูกค้าทั้งหมด: {len(customer_ids)} ราย | จำนวนสินค้าทั้งหมด: {len(product_lookup)} ราย")
st.caption("พัฒนาโดย: โปรเจกต์วิจัยการแนะนำสินค้าด้วยวิธี Matrix Factorization")
