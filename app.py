import streamlit as st
import cv2
import numpy as np
import pytesseract

# ==========================================
# CẤU HÌNH & HÀM PHỤ TRỢ
# ==========================================

def sort_contours_grid(cnts, row_sensitivity=15):
    """Sắp xếp contour theo thứ tự từ Trái -> Phải, Trên -> Dưới"""
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    c_boxes = list(zip(cnts, boundingBoxes))
    
    # Sắp xếp theo trục Y (để phân dòng)
    c_boxes.sort(key=lambda b: b[1][1]) 

    rows = []
    current_row = []
    last_y = -999

    for c, box in c_boxes:
        y = box[1]
        # Nếu box này nằm cùng dòng với box trước (sai số Y nhỏ)
        if y - last_y < row_sensitivity and last_y != -999:
            current_row.append((c, box))
        else:
            if current_row:
                # Sắp xếp dòng cũ theo trục X (Trái -> Phải)
                current_row.sort(key=lambda b: b[1][0])
                rows.extend(current_row)
            current_row = [(c, box)]
            last_y = y
    
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        rows.extend(current_row)

    return [item[0] for item in rows]

def preprocess_roi(roi):
    """
    Xử lý ảnh cắt (ROI) để làm sạch nhiễu nền
    """
    # 1. Phóng to ảnh (Upscale) để chữ rõ hơn
    roi = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # 2. Threshold OTSU (Chữ đen/Nền trắng)
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Padding (Thêm viền trắng) để chữ không dính mép
    thresh = cv2.copyMakeBorder(thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    
    return thresh

def process_image(image_file):
    # Đọc ảnh từ upload
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Chuyển xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tách nền để tìm ô Button màu trắng
    # Dùng HSV để bắt màu trắng tốt hơn RGB
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160]) # Giảm ngưỡng Sáng (Value)
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Morph Open để xóa nhiễu nhỏ, giữ lại khối button hình chữ nhật
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Tìm contours
    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    img_h, img_w = img.shape[:2]

    # Lọc các contour hợp lệ (kích thước giống ô code)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Điều kiện kích thước (Area > 800, hình chữ nhật ngang)
        if area > 800 and 2.0 < aspect_ratio < 8.0 and w < (img_w * 0.5):
            # Kiểm tra độ sáng trung bình vùng đó (tránh lấy nhầm vùng tối)
            roi_check = gray[y:y+h, x:x+w]
            if cv2.mean(roi_check)[0] > 150: # Phải là nền sáng
                valid_contours.append(c)

    detected_codes = []
    
    if valid_contours:
        # Sắp xếp đúng thứ tự
        valid_contours = sort_contours_grid(valid_contours, row_sensitivity=20)
        
        for idx, c in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(c)
            
            # Cắt ảnh (Crop) - Thu nhỏ vùng cắt một chút để loại bỏ viền đen của button
            pad_x = int(w * 0.08)
            pad_y = int(h * 0.15)
            roi = gray[y+pad_y : y+h-pad_y, x+pad_x : x+w-pad_x]
            
            if roi.size == 0: continue
            
            try:
                # Xử lý ảnh trước khi đọc
                processed_roi = preprocess_roi(roi)
                
                # --- CẤU HÌNH TESSERACT QUAN TRỌNG ---
                # psm 7: Coi là 1 dòng văn bản duy nhất
                # whitelist: CHỈ CHO PHÉP ĐỌC A-Z VÀ 0-9. (Sẽ tự loại bỏ dấu ~, |, -)
                config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                
                text = pytesseract.image_to_string(processed_roi, config=config)
                
                # Xóa khoảng trắng thừa
                final_code = text.strip().replace(" ", "")
                
                # --- LOGIC CHỐT ĐỘ DÀI 6 KÝ TỰ ---
                # Nếu dài hơn 6 (do nhiễu), cắt lấy 6 ký tự đầu
                if len(final_code) > 6:
                    final_code = final_code[:6]
                
                # Chỉ lấy code nếu độ dài từ 5 đến 6 ký tự
                if 5 <= len(final_code) <= 6:
                    detected_codes.append(final_code)
                    
                    # Vẽ lên ảnh để check
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, final_code, (x, y + h - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                
            except Exception as e:
                continue

    return img, detected_codes

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Tool Scan Code OKVIP (Final)", layout="wide")
st.title("🧩 Tool Quét Code - Fix Lỗi 6 Ký Tự")

uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    processed_img, codes = process_image(uploaded_file)
    
    with col1:
        st.subheader("Ảnh đã nhận diện")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("Kết quả (Copy)")
        if codes:
            txt = "\n".join(codes)
            st.text_area("Danh sách code:", value=txt, height=500)
            st.success(f"Đã tìm thấy {len(codes)} mã.")
        else:
            st.error("Không tìm thấy mã nào. Vui lòng thử ảnh rõ nét hơn.")
