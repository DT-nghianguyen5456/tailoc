import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# ==========================================
# CẤU HÌNH TESSERACT (CHỈ DÀNH CHO WINDOWS)
# Nếu bạn dùng Linux/Mac hoặc đã thêm vào PATH thì bỏ qua dòng này
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ==========================================

def clean_text(text):
    """
    Hàm lọc text: Chuyển thành chữ in hoa và xóa hết ký tự đặc biệt.
    Ví dụ: '9.B~4~U|J,D' -> '9B4UJD'
    """
    # Chỉ giữ lại ký tự chữ (a-z, A-Z) và số (0-9)
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned.upper()

def sort_contours(cnts, method="left-to-right"):
    """
    Hàm sắp xếp vị trí các ô để đọc theo thứ tự từ trái qua phải, trên xuống dưới.
    """
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def process_image(image_file):
    # Đọc ảnh từ file upload
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Phân ngưỡng (Threshold)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_codes = []
    valid_contours = []

    # Lọc contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Điều kiện lọc
        if area > 2000 and aspect_ratio > 2.0:
            valid_contours.append(c)

    # Sắp xếp và xử lý
    if valid_contours:
        (valid_contours, _) = sort_contours(valid_contours, method="top-to-bottom")
        
        for c in valid_contours:
            x, y, w, h = cv2.boundingRect(c)
            
            # --- SỬA LỖI Ở ĐÂY: Xử lý cắt ảnh an toàn hơn ---
            # Đảm bảo không cắt lẹm vào quá sâu khiến ảnh bị rỗng
            pad = 5
            # Kiểm tra nếu ô quá nhỏ thì không trừ margin nữa
            if h <= 2*pad or w <= 2*pad:
                roi = img[y:y+h, x:x+w]
            else:
                roi = img[y+pad:y+h-pad, x+pad:x+w-pad] 
            
            # --- KIỂM TRA QUAN TRỌNG ---
            # Nếu roi rỗng (size = 0) thì bỏ qua ngay, không đưa vào cvtColor
            if roi.size == 0:
                continue
            
            try:
                # Xử lý ảnh con
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Dùng Tesseract
                text = pytesseract.image_to_string(roi_thresh, config='--psm 6')
                cleaned = clean_text(text)
                
                if cleaned:
                    detected_codes.append(cleaned)
                    # Vẽ khung xanh
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(img, cleaned, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                # Nếu lỗi ở một ô nào đó, in ra console và tiếp tục ô khác chứ không dừng app
                print(f"Lỗi xử lý 1 ô: {e}")
                continue

    return img, detected_codes

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Trích xuất Mã Code", layout="wide")

st.title("🧩 Công cụ Trích xuất & Lọc Mã Code")
st.markdown("Tải ảnh lên để nhận diện các ô trắng, lọc ký tự đặc biệt và lấy mã code.")

uploaded_file = st.file_uploader("Chọn ảnh chứa mã code...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Ảnh gốc & Nhận diện")
        processed_img, codes = process_image(uploaded_file)
        # Chuyển đổi màu BGR sang RGB để hiển thị đúng trên Streamlit
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Các ô đã nhận diện (Khung xanh)", use_container_width=True)

    with col2:
        st.subheader("Kết quả Code đã lọc")
        if codes:
            st.success(f"Tìm thấy {len(codes)} mã code.")
            st.markdown("---")
            for idx, code in enumerate(codes):
                # Hiển thị từng code kèm nút copy
                st.markdown(f"**Code #{idx+1}**")
                st.code(code, language="text")
        else:
            st.warning("Không tìm thấy mã nào. Hãy thử ảnh rõ nét hơn.")

