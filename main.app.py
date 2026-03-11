import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. ตั้งค่าหน้าตาของเว็บ (Layout)
st.set_page_config(page_title="Retail Smart AI - NMF Dashboard", layout="wide")

# 2. โหลดข้อมูล
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

# 4. ส่วนแถบด้านข้าง (Sidebar)
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    
    selected_customer_id = st.selectbox(
        "เลือกรหัสลูกค้า (Customer ID):", 
        options=customer_ids,
        help="พิมพ์ค้นหารหัสลูกค้าในช่องนี้ได้เลย"
    )
    
    user_idx = customer_ids.index(selected_customer_id)
    top_n = st.slider("จำนวนสินค้าแนะนำต่อราย:", 1, 10, 5)
    
    st.divider()
    st.info(f"💡 ข้อมูลประมวลผลจาก:\n- 40 คุณลักษณะแฝง (K)\n- รหัสลูกค้าปัจจุบัน: {selected_customer_id}")

# 5. ส่วนแสดงผลหลัก
tab1, tab2 = st.tabs(["👤 การแนะนำสินค้าใหม่รายบุคคล", "📊 วิเคราะห์ 40 คุณลักษณะแฝง"])

with tab1:
    if st.button('🚀 เริ่มประมวลผลการแนะนำ'):
        # --- 1. คำนวณคะแนนทำนาย (Matrix Multiplication) ---
        user_scores = P[user_idx].dot(Q)
        
        # --- 2. ดึงประวัติการซื้อเดิม (เพื่อนำไปกรองออก) ---
        # ตรวจสอบว่าเป็น Sparse Matrix หรือไม่
        if hasattr(R_final, 'toarray'):
            user_history = R_final[user_idx].toarray().flatten()
        else:
            user_history = R_final[user_idx]
            
        # --- 3. สร้างตารางเพื่อกรองสินค้าที่เคยซื้อแล้ว ---
        rec_df = pd.DataFrame({
            'item_idx': np.arange(len(user_scores)),
            'score': user_scores,
            'already_bought': user_history > 0
        })
        
        # --- 4. กรองสินค้าที่เคยซื้อออก (เอาเฉพาะ False) และจัดลำดับคะแนนจากมากไปน้อย ---
        new_items_only = rec_df[rec_df['already_bought'] == False]
        recommended_indices = new_items_only.sort_values(by='score', ascending=False).head(top_n)['item_idx'].values
        
        # --- 5. แสดงวิเคราะห์พฤติกรรม (Latent Features) ---
        user_features = P[user_idx]
        top_k_indices = user_features.argsort()[-2:][::-1]
        
        st.subheader(f"🔍 วิเคราะห์พฤติกรรมรหัสลูกค้า: {selected_customer_id}")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.success(f"📌 **กลุ่มหลัก:** คุณลักษณะแฝงที่ {top_k_indices[0] + 1}")
        with f_col2:
            st.success(f"📌 **กลุ่มรอง:** คุณลักษณะแฝงที่ {top_k_indices[1] + 1}")
        
        st.markdown("---")

        # --- 6. แสดงผลการแนะนำและประวัติ ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 สินค้าใหม่ที่คุณอาจสนใจ (ยังไม่เคยซื้อ)")
            if len(recommended_indices) == 0:
                st.write("ไม่พบสินค้าแนะนำใหม่")
            else:
                for i, idx in enumerate(recommended_indices):
                    row = product_lookup.iloc[idx]
                    st.write(f"**{i+1}. {row['product_name']}**")
                    st.caption(f"รหัสสินค้า: {row['product_id']} | คะแนนความน่าจะเป็น: {user_scores[idx]:.4f}")
                
        with col2:
            st.subheader("📜 ประวัติการซื้อเดิม (Top History)")
            past_indices = np.where(user_history > 0)[0]
            
            if len(past_indices) == 0:
                st.warning("ไม่พบประวัติการซื้อ")
            else:
                for idx in past_indices[:10]:
                    name = product_lookup.iloc[idx]['product_name']
                    st.write(f"✅ {name}")

with tab2:
    st.subheader("📊 รายละเอียดคุณลักษณะแฝง (Latent Feature Catalog)")
    st.markdown("วิเคราะห์สินค้าเด่น 10 อันดับแรกในแต่ละกลุ่มพฤติกรรม (K)")
    
    selected_k = st.number_input("ระบุลำดับคุณลักษณะแฝง (Index K):", 1, 40, 1)
    
    # ดึงน้ำหนักสินค้าจาก Matrix Q
    try:
        feature_weights = Q[selected_k - 1] 
    except:
        feature_weights = Q[:, selected_k - 1]

    top_item_idx = feature_weights.argsort()[-10:][::-1]
    
    topic_data = []
    for idx in top_item_idx:
        item_name = product_lookup.iloc[idx]['product_name']
        p_id = product_lookup.iloc[idx]['product_id']
        weight = feature_weights[idx]
        topic_data.append({
            "รหัสสินค้า": p_id,
            "ชื่อสินค้า": item_name,
            "ค่าน้ำหนัก": f"{weight:.4f}"
        })
    
    st.table(pd.DataFrame(topic_data))

# 6. ส่วนท้าย (Footer)
st.markdown("---")
st.caption(f"สถานะ: พร้อมใช้งาน | กรองสินค้าที่เคยซื้อออก: เปิดใช้งาน ✅ | ฐานข้อมูลลูกค้า: {len(customer_ids)} ราย")
