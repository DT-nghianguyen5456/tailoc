import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

st.set_page_config(page_title="Tool Lọc Code OKVIP", page_icon="⚡")

# Hàm làm sạch: Xóa mọi ký tự đặc biệt, chỉ giữ Chữ và Số
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def process_image(image_file):
    # 1. Đọc ảnh từ upload
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 2. Xử lý ảnh để tìm khung (Pre-processing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Nhị phân hóa: Lấy vùng màu trắng sáng (>180)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # --- KỸ THUẬT QUAN TRỌNG: MORPHOLOGICAL CLOSING ---
    # Lệnh này giúp "hàn gắn" các chữ đen bên trong ô trắng.
    # Biến cả ô code thành 1 khối hình chữ nhật đặc màu trắng.
    # Giúp giảm số lượng contour từ 1300 xuống còn đúng số lượng ô code (khoảng 20).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Tìm viền
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # BỘ LỌC KÍCH THƯỚC:
        # - w > h: Ô code nằm ngang
        # - w > 50: Chiều rộng phải đủ lớn (tránh nhiễu)
        # - area > 1000: Diện tích phải lớn
        if w > h and w > 50 and h > 20 and area > 1000:
            valid_boxes.append((x, y, w, h))
            
    # --- SAFETY LOCK (CHỐNG TREO MÁY) ---
    # Chỉ lấy tối đa 20 ô có diện tích lớn nhất.
    # Đảm bảo dù ảnh nhiễu đến đâu cũng không bao giờ bị treo.
    if len(valid_boxes) > 20:
        valid_boxes = sorted(valid_boxes, key=lambda b: b[2]*b[3], reverse=True)[:20]
    
    # Sắp xếp các ô từ trên xuống dưới, trái qua phải
    valid_boxes.sort(key=lambda b: (b[1] // 40, b[0])) 

    results = []
    
    # 3. Đọc OCR từng ô
    for (x, y, w, h) in valid_boxes:
        # Cắt vùng ảnh gốc (lấy từ ảnh gray để rõ nét)
        roi = gray[y:y+h, x:x+w]
        
        # Phóng to ảnh lên 2 lần để đọc chữ rõ hơn
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Otsu Threshold để tách chữ đen trên nền trắng tuyệt đối
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Thêm viền trắng xung quanh (Padding) để Tesseract không bị mất chữ sát lề
        roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255])
        
        # Cấu hình Tesseract:
        # --psm 7: Coi là 1 dòng đơn
        # whitelist: Chỉ cho phép nhận diện A-Z và 0-9
        config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(roi, config=config)
        
        # Dùng thêm Python Regex để lọc sạch lần cuối
        cleaned = clean_text(text)
        
        # Chỉ lấy kết quả nếu dài hơn 3 ký tự
        if len(cleaned) > 3:
            results.append(cleaned)
            
    return results

# --- GIAO DIỆN WEB ---
st.title("⚡ Tool Quét Code OKVIP (Bản V5)")
st.info("Đã sửa lỗi treo máy và tối ưu nhận diện ký tự đặc biệt.")

uploaded_file = st.file_uploader("Chọn ảnh để quét...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Ảnh đã tải lên', use_container_width=True)
    
    if st.button('🚀 Bắt đầu quét ngay'):
        with st.spinner('Đang xử lý hình ảnh...'):
            try:
                codes = process_image(uploaded_file)
                
                if codes:
                    st.success(f"Hoàn tất! Tìm thấy {len(codes)} mã code.")
                    st.markdown("---")
                    
                    # Hiển thị kết quả dạng lưới 2 cột
                    col1, col2 = st.columns(2)
                    for i, code in enumerate(codes):
                        # Chia cột hiển thị
                        if i % 2 == 0:
                            with col1:
                                st.code(code, language=None)
                        else:
                            with col2:
                                st.code(code, language=None)
                else:
                    st.warning("Không tìm thấy mã nào hợp lệ. Hãy thử cắt ảnh gọn hơn.")
                    
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
