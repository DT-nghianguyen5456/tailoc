import streamlit as st
import cv2
import numpy as np
import pytesseract
import re

# ==========================================
# CẤU HÌNH XỬ LÝ ẢNH
# ==========================================

def clean_text(text):
    """Giữ lại chữ và số, viết hoa."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

def sort_contours_grid(cnts, row_sensitivity=10):
    """Sắp xếp contour theo lưới (Trái->Phải, Trên->Dưới)"""
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    c_boxes = list(zip(cnts, boundingBoxes))
    
    # Sắp xếp theo Y (chiều dọc)
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
                current_row.sort(key=lambda b: b[1][0]) # Sắp xếp hàng cũ theo X
                rows.extend(current_row)
            current_row = [(c, box)]
            last_y = y
    
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        rows.extend(current_row)

    return [item[0] for item in rows]

def reconstruct_clean_image(roi):
    """
    Tách các ký tự chính trong ô và vẽ lại lên nền trắng sạch.
    Mục tiêu: Loại bỏ hoàn toàn dấu ~ . , | _
    """
    # 1. Chuyển xám và phân ngưỡng
    # Dùng ngưỡng cố định vì nền đã chắc chắn là trắng
    _, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)

    # 2. Tìm contours bên trong ô
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Danh sách chứa (x, contour)
    valid_chars = []
    h_roi, w_roi = roi.shape[:2]
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        
        # --- BỘ LỌC KÝ TỰ RÁC ---
        # 1. Chiều cao: Chữ cái phải cao ít nhất 35% chiều cao ô (loại bỏ . , - ~)
        if h < h_roi * 0.35: continue
            
        # 2. Chiều rộng:
        # - Phải đủ rộng (> 4px) để loại bỏ gạch đứng | hoặc nhiễu
        # - Không được quá rộng (> 80% ô) để loại bỏ viền dính
        if w < 4 or w > w_roi * 0.8: continue
        
        # 3. Diện tích: Phải đủ lớn
        if w * h < 50: continue

        valid_chars.append((x, c))

    # Sắp xếp theo thứ tự trái sang phải
    valid_chars.sort(key=lambda k: k[0])
    
    # Giới hạn lấy tối đa 6 ký tự (nếu bộ lọc vẫn sót)
    # Thường thì bộ lọc chiều cao đã loại hết rác rồi
    final_chars = [item[1] for item in valid_chars[:6]]

    # 3. Vẽ lại ảnh sạch
    clean_img = np.ones((h_roi, w_roi), dtype=np.uint8) * 255 # Nền trắng
    cv2.drawContours(clean_img, final_chars, -1, 0, -1) # Vẽ chữ màu đen
    
    # Thêm viền trắng bao quanh cho Tesseract dễ đọc
    clean_img = cv2.copyMakeBorder(clean_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    
    return clean_img

def process_image(image_file):
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # --- BƯỚC 1: LỌC MÀU (CHỈ LẤY MÀU TRẮNG) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa màu trắng trong HSV:
    # Saturation thấp (0-40), Value cao (200-255)
    lower_white = np.array([0, 0, 215])
    upper_white = np.array([180, 40, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Làm sạch mask (xóa nhiễu li ti)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2) # Nở vùng trắng ra chút để contour liền mạch

    # --- BƯỚC 2: TÌM KHUNG Ô ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    boxes_info = [] # Lưu (area, contour) để tính toán thống kê
    
    img_h, img_w = img.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Lọc thô: 
        # - Diện tích > 1500 (bỏ chữ nhỏ rời rạc)
        # - Tỷ lệ 2.0 < w/h < 6.0 (hình chữ nhật ngang)
        # - Chiều rộng < 1/3 ảnh (loại bỏ banner dài ngoằng)
        if area > 1500 and 2.0 < aspect_ratio < 6.0 and w < (img_w * 0.4):
            valid_contours.append(c)
            boxes_info.append(area)

    # --- BƯỚC 3: LỌC NGOẠI LAI (OUTLIERS) ---
    # Các ô code thường có diện tích xấp xỉ nhau.
    # Nếu có 1 ô quá to (banner Telegram còn sót) hoặc quá nhỏ, ta loại nó.
    final_contours = []
    if boxes_info:
        median_area = np.median(boxes_info)
        # Chỉ giữ lại các ô có diện tích lệch không quá 40% so với trung bình
        for c in valid_contours:
            area = cv2.contourArea(c)
            if 0.6 * median_area < area < 1.4 * median_area:
                final_contours.append(c)
    else:
        final_contours = valid_contours

    detected_codes = []

    if final_contours:
        # Sắp xếp contour
        final_contours = sort_contours_grid(final_contours, row_sensitivity=20)
        
        # Chuyển ảnh gốc sang xám để cắt (crop)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for idx, c in enumerate(final_contours):
            x, y, w, h = cv2.boundingRect(c)
            
            # Cắt ảnh (Crop) - Thụt vào trong (padding) 4px để bỏ viền lờ mờ
            pad = 4
            roi = gray[y+pad:y+h-pad, x+pad:x+w-pad]
            
            if roi.size == 0: continue

            # --- BƯỚC 4: TÁI TẠO & OCR ---
            try:
                clean_roi = reconstruct_clean_image(roi)
                
                # Cấu hình Tesseract chỉ nhận chữ số và chữ cái
                config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(clean_roi, config=config)
                final_code = clean_text(text)
                
                # Xử lý kết quả
                if len(final_code) > 6: final_code = final_code[:6] # Cắt thừa
                
                detected_codes.append(final_code)
                
                # Vẽ lên ảnh (chỉ để hiển thị)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, final_code, (x, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception:
                continue

    return img, detected_codes, mask

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="OKVIP Code Extractor v3", layout="wide")

st.title("🧩 Tool Quét Code - Chế độ Lọc Màu Trắng")
st.markdown("**Cập nhật:** Sử dụng bộ lọc màu HSV để chỉ bắt các ô trắng tinh, loại bỏ hoàn toàn banner Telegram và khung viền vàng.")

uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    processed_img, codes, debug_mask = process_image(uploaded_file)
    
    with col1:
        st.subheader("Kết quả trên Ảnh gốc")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with st.expander("Xem Chế độ nhìn của AI (Debug Mask)"):
            st.image(debug_mask, caption="Những vùng màu trắng là vùng AI nhìn thấy", use_container_width=True)

    with col2:
        st.subheader("Danh sách Code")
        if codes:
            txt_output = "\n".join(codes)
            st.text_area("Copy Code:", value=txt_output, height=400)
            st.success(f"Tìm thấy {len(codes)} mã.")
        else:
            st.warning("Không tìm thấy mã nào. Hãy chắc chắn ảnh đủ sáng.")
