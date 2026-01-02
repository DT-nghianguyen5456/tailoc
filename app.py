import streamlit as st
import cv2
import numpy as np
import pytesseract
import re

# ==========================================
# CẤU HÌNH & HÀM PHỤ TRỢ
# ==========================================

def clean_text(text):
    """Giữ lại chữ và số, viết hoa."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

def sort_contours_grid(cnts, row_sensitivity=10):
    """Sắp xếp contour theo lưới (Trái->Phải, Trên->Dưới)"""
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    c_boxes = list(zip(cnts, boundingBoxes))
    c_boxes.sort(key=lambda b: b[1][1]) # Sắp xếp theo Y trước

    rows = []
    current_row = []
    last_y = -999

    for c, box in c_boxes:
        y = box[1]
        if y - last_y < row_sensitivity and last_y != -999:
            current_row.append((c, box))
        else:
            if current_row:
                current_row.sort(key=lambda b: b[1][0]) # Sắp xếp theo X
                rows.extend(current_row)
            current_row = [(c, box)]
            last_y = y
    
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        rows.extend(current_row)

    return [item[0] for item in rows]

def get_clean_6_chars_image(roi_gray):
    """
    CHIẾN THUẬT "TOP 6":
    1. Tìm tất cả contours trong ô.
    2. Chỉ giữ lại 6 contours có DIỆN TÍCH LỚN NHẤT.
    3. Vẽ lại 6 contours này lên nền trắng mới tinh để OCR.
    """
    # Threshold OTSU để tách chữ đen trên nền trắng
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tìm contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts: return roi_gray

    # Lưu danh sách (Diện tích, Contour)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10: # Bỏ nhiễu cực nhỏ
            candidates.append((area, c))
    
    # Sắp xếp theo diện tích giảm dần (Lớn nhất đứng đầu)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Lấy Top 6 (hoặc ít hơn nếu không đủ 6)
    top_6 = candidates[:6]
    
    # Sắp xếp 6 contour này theo vị trí X (Trái -> Phải) để code đúng thứ tự
    # item[1] là contour -> tính bounding rect của nó để lấy x
    top_6_sorted_x = sorted(top_6, key=lambda item: cv2.boundingRect(item[1])[0])
    
    # Vẽ lại ảnh sạch
    h, w = roi_gray.shape
    clean_img = np.ones((h, w), dtype=np.uint8) * 255 # Nền trắng
    
    final_cnts = [item[1] for item in top_6_sorted_x]
    cv2.drawContours(clean_img, final_cnts, -1, 0, -1) # Vẽ chữ màu đen
    
    # Thêm viền trắng an toàn
    clean_img = cv2.copyMakeBorder(clean_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    
    return clean_img

def process_image(image_file):
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. Xử lý ảnh để tìm khung
    # Chuyển sang ảnh xám để check độ sáng
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dùng HSV để bắt màu trắng (cho mask ban đầu)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180]) # Giảm ngưỡng Value xuống chút để bắt chắc
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # QUAN TRỌNG: Dùng Morph Open với kernel LỚN để xóa chữ mảnh, chỉ giữ khối button đặc
    kernel_size = 11 # Kích thước 11x11 sẽ xóa sạch chữ mảnh
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Tìm contours trên mask đã làm sạch
    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    img_h, img_w = img.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)
        
        # 1. Lọc kích thước hình học
        if area > 1000 and 2.0 < aspect_ratio < 6.0 and w < (img_w * 0.5):
            
            # 2. CHỐT CHẶN: KIỂM TRA ĐỘ SÁNG NỀN (MEAN BRIGHTNESS)
            # Cắt vùng ảnh xám
            roi_check = gray[y:y+h, x:x+w]
            mean_val = cv2.mean(roi_check)[0]
            
            # Button thật sự là nền trắng -> Mean phải rất cao (> 180)
            # Box "Telegram" nền tối -> Mean sẽ thấp (< 100) -> BỊ LOẠI
            if mean_val > 180:
                valid_contours.append(c)

    # Xử lý các ô hợp lệ
    detected_codes = []
    
    if valid_contours:
        # Sắp xếp thứ tự
        valid_contours = sort_contours_grid(valid_contours, row_sensitivity=20)
        
        for idx, c in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(c)
            
            # Crop
            pad = 5
            roi = gray[y+pad:y+h-pad, x+pad:x+w-pad]
            
            if roi.size == 0: continue
            
            try:
                # --- DÙNG THUẬT TOÁN TOP 6 ---
                clean_roi = get_clean_6_chars_image(roi)
                
                # OCR
                config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(clean_roi, config=config)
                final_code = clean_text(text)
                
                if final_code:
                    detected_codes.append(final_code)
                    
                    # Vẽ kết quả
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, final_code, (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            except Exception:
                continue

    return img, detected_codes, mask_clean

# --- GIAO DIỆN ---
st.set_page_config(page_title="OKVIP Fix Final", layout="wide")
st.title("🧩 Tool Quét Code - Fix Lỗi Nhận Diện")
st.markdown("Chế độ: **Top 6 Ký Tự** + **Kiểm Tra Độ Sáng Nền** (Loại bỏ box Telegram/Quà tặng)")

uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    processed_img, codes, debug_mask = process_image(uploaded_file)
    
    with col1:
        st.subheader("Kết quả")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with st.expander("Debug: Mask Button (Đã lọc chữ)"):
            st.image(debug_mask, use_container_width=True)

    with col2:
        st.subheader("Copy Code")
        if codes:
            txt = "\n".join(codes)
            st.text_area("Danh sách:", value=txt, height=400)
            st.success(f"Tìm thấy {len(codes)} mã.")
        else:
            st.warning("Không tìm thấy mã.")
