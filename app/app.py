# app/app.py

import streamlit as st
import requests
import time

st.set_page_config(page_title="VN Legal Chat", layout="centered")

st.title("⚖️ Vietnamese Legal Assistant")
st.caption("Dùng mô hình Meta LLaMA 3.1 8B để trả lời các câu hỏi pháp lý.")

# Giao diện nhập prompt
user_input = st.text_area("💬 Câu hỏi của bạn:", height=150)

if st.button("📨 Gửi câu hỏi") and user_input.strip():
    with st.spinner("⏳ Đang xử lý..."):
        start_time = time.time()
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"prompt": user_input},
                timeout=300  # tránh treo nếu model lâu
            )
            latency = time.time() - start_time

            if response.status_code == 200:
                result = response.json()["response"]
                st.success("🤖 Trợ lý pháp lý trả lời:")
                st.markdown(result)
                st.caption(f"⏱️ Thời gian phản hồi: {latency:.2f} giây")
            else:
                st.error(f"❌ Lỗi server: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Không thể kết nối tới API: {e}")
