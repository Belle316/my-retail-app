import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. ตั้งค่าหน้าตาของเว็บ (Layout)
st.set_page_config(page_title="Retail Smart AI - NMF Dashboard", layout="wide")

# 2. โหลดข้อมูล (Caching เพื่อให้โหลดเร็วขึ้น)
@st.cache_resource
def load_all_data():
    P = joblib.load('P_matrix.joblib')
    Q = joblib.load('Q_matrix.joblib')
    product_lookup = joblib.load('product_lookup.joblib')
    R_final = joblib.load('R_final.joblib')
    customer_ids = joblib.load('customer_ids.joblib') 
    return P, Q, product_lookup, R_final, customer_ids

P, Q, product_lookup, R_final, customer_ids = load_all_data()

# 3. ส่วนหัวของหน้าเว็บ
st.title("🛒 ระบบแนะนำสินค้าและวิเคราะห์คุณลักษณะแฝง (NMF)")
st.markdown("---")

# 4. ส่วนแถบด้านข้าง (Sidebar) สำหรับเลือกข้อมูล
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    
    selected_customer_id = st.selectbox(
        "เลือกรหัสลูกค้า (Customer ID):", 
        options=customer_ids,
        help="พิมพ์ค้นหารหัสลูกค้าในช่องนี้ได้เลย"
    )
    
    # แปลงรหัสลูกค้าเป็น Index เพื่อใช้กับ Matrix P
    user_idx = customer_ids.index(selected_customer_id)
    
    top_n = st.slider("จำนวนสินค้าแนะนำต่อราย:", 1, 10, 5)
    
    st.divider()
    st.info(f"💡 ข้อมูลประมวลผลจาก:\n- 40 คุณลักษณะแฝง (K)\n- รหัสลูกค้าปัจจุบัน: {selected_customer_id}")

# 5. ส่วนแสดงผลหลัก
# สร้าง Tabs เพื่อแยกมุมมองลูกค้า และ มุมมองเจ้าของร้าน (40 Topics)
tab1, tab2 = st.tabs(["👤 การแนะนำรายบุคคล", "📊 วิเคราะห์ 40 คุณลักษณะแฝง (Admin)"])

with tab1:
    if st.button('🚀 เริ่มประมวลผลการแนะนำ'):
        # --- คำนวณหาคะแนนการแนะนำ (Matrix Multiplication) ---
        # สูตร: R_pred = P[user] * Q.T (ในที่นี้ Q ของเราน่าจะเป็นรูป K x Items)
        user_scores = P[user_idx].dot(Q)
        
        # ดึงสินค้าที่ได้คะแนนสูงสุด Top-N
        recommended_indices = user_scores.argsort()[-top_n:][::-1]
        
        # --- แสดงกลุ่มพฤติกรรม (Latent Features) ของลูกค้าคนนี้ ---
        user_features = P[user_idx]
        top_k_indices = user_features.argsort()[-2:][::-1] # ดึง 2 กลุ่มที่เกี่ยวข้องที่สุด
        
        st.subheader(f"🔍 วิเคราะห์พฤติกรรมรหัสลูกค้า: {selected_customer_id}")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.success(f"📌 **กลุ่มหลัก:** คุณลักษณะแฝงที่ {top_k_indices[0] + 1}")
        with f_col2:
            st.success(f"📌 **กลุ่มรอง:** คุณลักษณะแฝงที่ {top_k_indices[1] + 1}")
        
        st.markdown("---")

        # --- แสดงผลการแนะนำและประวัติ ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 สินค้าที่แนะนำสำหรับคุณ")
            for i, idx in enumerate(recommended_indices):
                row = product_lookup.iloc[idx]
                st.write(f"**{i+1}. {row['product_name']}**")
                st.caption(f"รหัสสินค้า: {row['product_id']}")
                
        with col2:
            st.subheader("📜 ประวัติการซื้อเดิม (Top History)")
            user_history = R_final[user_idx].toarray().flatten()
            past_indices = np.where(user_history > 0)[0]
            
            if len(past_indices) == 0:
                st.warning("ไม่พบประวัติการซื้อ")
            else:
                # แสดงประวัติการซื้อ (จำกัดที่ 10 รายการ)
                for idx in past_indices[:10]:
                    name = product_lookup.iloc[idx]['product_name']
                    st.write(f"✅ {name}")

with tab2:
    st.subheader("📊 รายละเอียดคุณลักษณะแฝง (Latent Feature Catalog)")
    st.markdown("วิเคราะห์สินค้าเด่น 10 อันดับแรกในแต่ละกลุ่มพฤติกรรม (K) เพื่อจัดโปรโมชั่น")
    
    # เลือก Topic (K) ที่ต้องการดู
    selected_k = st.number_input("ระบุลำดับคุณลักษณะแฝ5 (Index K):", 1, 40, 1)
    
    # ดึงค่าน้ำหนักสินค้าใน Topic นั้น (จาก Matrix Q)
    # หมายเหตุ: ตรวจสอบแกนของ Q (หากเป็น K x Items ให้ใช้ Q[selected_k-1])
    # หาก Q เป็น Items x K ให้ใช้ Q[:, selected_k-1]
    try:
        # สมมติฐาน: Q มีขนาด (40, จำนวนสินค้า)
        feature_weights = Q[selected_k - 1] 
    except:
        # กรณี Q มีขนาด (จำนวนสินค้า, 40)
        feature_weights = Q[:, selected_k - 1]

    # หา 10 สินค้าที่มีค่าน้ำหนักสูงสุดในกลุ่มนั้น
    top_item_idx = feature_weights.argsort()[-10:][::-1]
    
    # สร้างตารางแสดงผล
    topic_data = []
    for idx in top_item_idx:
        item_name = product_lookup.iloc[idx]['product_name']
        p_id = product_lookup.iloc[idx]['product_id']
        weight = feature_weights[idx]
        topic_data.append({
            "รหัสสินค้า": p_id,
            "ชื่อสินค้า": item_name,
            "ค่าน้ำหนักความถี่": f"{weight:.4f}"
        })
    
    st.table(pd.DataFrame(topic_data))
    st.info(f"💡 สินค้ากลุ่มนี้คือตัวแทนของคุณลักษณะแฝงที่ {selected_k} ซึ่งถูกคำนวณจากรูปแบบความถี่การซื้อซ้ำในระบบ")

# 6. ส่วนท้าย (Footer)
st.markdown("---")
st.caption(f"สถานะ: พร้อมใช้งาน | ฐานข้อมูลลูกค้า: {len(customer_ids)} ราย | ฐานข้อมูลสินค้า: {len(product_lookup)} ราย")
st.write("--- ทดสอบการอัปเดตโค้ด: เวอร์ชันใหม่ทำงานแล้ว ---")
