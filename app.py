import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# ==========================================
# CẤU HÌNH
# ==========================================

def clean_text(text):
    """
    Lọc text: Chỉ giữ lại chữ cái và số, viết hoa toàn bộ.
    VD: '9.B~4~U|J,D' -> '9B4UJD'
    """
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned.upper()

def sort_contours_grid(cnts, max_cols=4, row_sensitivity=20):
    """
    Sắp xếp contour theo dạng lưới (Grid):
    - Gom nhóm các contour có vị trí Y gần nhau (cùng 1 hàng).
    - Trong mỗi hàng, sắp xếp theo vị trí X (từ trái qua phải).
    """
    # Lấy bounding rect cho tất cả contours
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    # Zip contour và box lại với nhau
    c_boxes = list(zip(cnts, boundingBoxes))
    
    # Sắp xếp sơ bộ theo chiều Y (từ trên xuống dưới)
    c_boxes.sort(key=lambda b: b[1][1])

    # Gom nhóm theo hàng
    rows = []
    current_row = []
    last_y = -999

    for c, box in c_boxes:
        y = box[1]
        # Nếu khoảng cách Y so với hàng trước < row_sensitivity thì coi như cùng hàng
        if y - last_y < row_sensitivity and last_y != -999:
            current_row.append((c, box))
        else:
            # Nếu lệch nhiều -> Hàng mới
            if current_row:
                # Sắp xếp hàng cũ theo X (trái qua phải) và lưu lại
                current_row.sort(key=lambda b: b[1][0])
                rows.extend(current_row)
            current_row = [(c, box)]
            last_y = y
    
    # Lưu hàng cuối cùng
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        rows.extend(current_row)

    # Tách lại thành list contours
    sorted_cnts = [item[0] for item in rows]
    return sorted_cnts

def process_image(image_file):
    # 1. Đọc ảnh
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. Tiền xử lý để tìm ô trắng
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dùng ngưỡng cao (220) để chỉ bắt các màu gần trắng, loại bỏ nền màu
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    
    # Morphological Close để làm liền các khối (phòng trường hợp chữ làm rách contour)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 3. Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    img_h, img_w = img.shape[:2]

    # 4. Lọc Contours (Chỉ lấy hình chữ nhật kích thước hợp lý)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Điều kiện lọc chặt chẽ hơn:
        # - Diện tích phải đủ lớn (>1000)
        # - Tỉ lệ khung hình (w/h) phải là hình chữ nhật ngang (2.0 < ratio < 6.0)
        # - Chiều rộng không được quá to (tránh banner tiêu đề) và không quá nhỏ
        if area > 1000 and 2.0 < aspect_ratio < 7.0 and w < (img_w * 0.9) and h < (img_h * 0.2):
            valid_contours.append(c)

    detected_codes = []

    # 5. Sắp xếp và OCR
    if valid_contours:
        # Sắp xếp theo dạng lưới (Trái->Phải, Trên->Dưới)
        valid_contours = sort_contours_grid(valid_contours, row_sensitivity=img_h//20)
        
        for idx, c in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(c)
            
            # Cắt ảnh (Crop) - Thụt vào trong 5px để loại bỏ viền đen/nhiễu của contour
            pad = 5
            roi = gray[y+pad:y+h-pad, x+pad:x+w-pad]
            
            if roi.size == 0: continue

            # --- TỐI ƯU HÓA ẢNH CHO TESSERACT ---
            # 1. Threshold cục bộ để chữ đen đậm, nền trắng tuyệt đối
            _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 2. Thêm viền trắng bao quanh (Padding) để Tesseract không bị lỗi sát viền
            roi_padded = cv2.copyMakeBorder(roi_thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255])

            # 3. OCR với config PSM 7 (Treat as single text line)
            text = pytesseract.image_to_string(roi_padded, config='--psm 7')
            cleaned = clean_text(text)
            
            # Logic kiểm tra: Code thường có ít nhất 4 ký tự
            if len(cleaned) >= 4:
                detected_codes.append(cleaned)
                
                # Vẽ lên ảnh kết quả
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Đánh số thứ tự để người dùng dễ đối chiếu
                cv2.putText(img, str(idx + 1), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    return img, detected_codes

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Trích xuất Code OKVIP", layout="wide")

st.title("🧩 Tool Quét Mã Code - Tự Động Lọc Ký Tự")
st.markdown("""
**Hướng dẫn:** 1. Tải ảnh chứa bảng code lên.
2. Hệ thống sẽ tự tìm ô màu trắng, đọc chữ và xóa ký tự đặc biệt.
3. Nhấn nút copy ở cột bên phải.
""")

uploaded_file = st.file_uploader("Tải ảnh lên (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🖼️ Ảnh đã nhận diện")
        processed_img, codes = process_image(uploaded_file)
        # Hiển thị ảnh
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader(f"📋 Kết quả ({len(codes)} mã)")
        if codes:
            # Tạo container cuộn nếu danh sách dài
            with st.container(height=600):
                for idx, code in enumerate(codes):
                    st.markdown(f"**Ô số {idx+1}:**")
                    st.code(code, language="text")
        else:
            st.warning("⚠️ Không tìm thấy mã nào. Vui lòng đảm bảo ảnh rõ nét và không bị chói sáng quá mức vào các ô chữ.")
