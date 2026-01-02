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
    """
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned.upper()

def sort_contours_grid(cnts, row_sensitivity=20):
    """
    Sắp xếp contour theo lưới: Hàng trên -> Hàng dưới, Trái -> Phải
    """
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    c_boxes = list(zip(cnts, boundingBoxes))
    
    # Sắp xếp theo Y trước
    c_boxes.sort(key=lambda b: b[1][1])

    rows = []
    current_row = []
    last_y = -999

    for c, box in c_boxes:
        y = box[1]
        if y - last_y < row_sensitivity and last_y != -999:
            current_row.append((c, box))
        else:
            if current_row:
                current_row.sort(key=lambda b: b[1][0])
                rows.extend(current_row)
            current_row = [(c, box)]
            last_y = y
    
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        rows.extend(current_row)

    return [item[0] for item in rows]

def get_clean_6_chars_image(roi_gray):
    """
    THUẬT TOÁN "TOP 6":
    1. Tìm tất cả contours trong ô.
    2. Chỉ lấy 6 contours có DIỆN TÍCH LỚN NHẤT (Chữ cái thật).
    3. Loại bỏ tất cả contours nhỏ (dấu chấm, phẩy, gạch mảnh).
    4. Vẽ lại 6 contours này lên nền trắng để Tesseract đọc.
    """
    # 1. Threshold để tách chữ
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Tìm contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return roi_gray

    # 3. Phân tích và lọc
    char_candidates = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Lọc nhiễu cơ bản (quá nhỏ thì bỏ qua luôn)
        if area < 20 or w < 3 or h < 8:
            continue
            
        char_candidates.append((area, x, y, w, h, c))

    # 4. CHIẾN THUẬT QUAN TRỌNG: Chỉ lấy Top 6 Area lớn nhất
    # (Vì chữ cái thật luôn to hơn dấu ~ . , |)
    char_candidates.sort(key=lambda x: x[0], reverse=True) # Sắp xếp diện tích giảm dần
    top_chars = char_candidates[:6] # Lấy 6 cái to nhất

    # 5. Sắp xếp lại 6 chữ cái này theo thứ tự Trái -> Phải (theo toạ độ x)
    top_chars.sort(key=lambda x: x[1]) 

    # 6. Vẽ lại ảnh mới sạch sẽ
    clean_img = np.ones_like(roi_gray) * 255 # Tạo nền trắng tinh
    
    # Vẽ các chữ cái đã chọn lên nền trắng (màu đen)
    for _, _, _, _, _, c in top_chars:
        cv2.drawContours(clean_img, [c], -1, 0, -1) # Vẽ đặc (thickness = -1)

    # Thêm viền trắng bao quanh ảnh để Tesseract dễ đọc
    clean_img = cv2.copyMakeBorder(clean_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255])
    
    return clean_img

def process_image(image_file):
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tìm các ô trắng
    _, thresh_bg = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_bg = cv2.morphologyEx(thresh_bg, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(thresh_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    img_h, img_w = img.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        if area > 1000 and 2.0 < aspect_ratio < 7.0 and w < (img_w * 0.9):
            valid_contours.append(c)

    detected_codes = []

    if valid_contours:
        valid_contours = sort_contours_grid(valid_contours, row_sensitivity=img_h//20)
        
        for idx, c in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(c)
            
            pad = 5
            if h > 2*pad and w > 2*pad:
                roi = gray[y+pad:y+h-pad, x+pad:x+w-pad]
            else:
                roi = gray[y:y+h, x:x+w]
            
            if roi.size == 0: continue

            # --- SỬ DỤNG THUẬT TOÁN MỚI ---
            # Chỉ lấy đúng 6 ký tự to nhất, vẽ lại ảnh mới
            clean_roi = get_clean_6_chars_image(roi)
            
            # OCR
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(clean_roi, config=config)
            final_code = clean_text(text)
            
            # Xử lý kết quả: Nếu > 6 ký tự (do Tesseract đọc nhầm nét đứt), cắt lấy 6 đầu
            # Nếu < 6, vẫn giữ nguyên
            if len(final_code) > 6:
                final_code = final_code[:6]

            detected_codes.append(final_code)
            
            # Vẽ lại lên ảnh gốc để hiển thị
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Hiển thị code ngay trên ảnh
            cv2.putText(img, final_code, (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return img, detected_codes

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Trích xuất Code OKVIP (6 Chars)", layout="wide")

st.title("🧩 Tool Quét Code - Chế độ 6 Ký Tự")
st.info("💡 Thuật toán mới: Tự động chọn 6 ký tự lớn nhất trong mỗi ô và loại bỏ hoàn toàn các ký tự rác (dấu chấm, dấu ngã, gạch đứng).")

uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Kết quả xử lý")
        processed_img, codes = process_image(uploaded_file)
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Danh sách Code")
        if codes:
            txt_output = ""
            for code in codes:
                txt_output += code + "\n"
            
            st.text_area("Copy tất cả ở đây:", value=txt_output, height=400)
            
            st.markdown("---")
            st.write(f"Đã tìm thấy **{len(codes)}** mã.")
        else:
            st.warning("Không tìm thấy mã nào.")
