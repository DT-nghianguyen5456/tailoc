import streamlit as st
import cv2
import numpy as np
import pytesseract
import re

# ==========================================
# CẤU HÌNH & HÀM PHỤ TRỢ
# ==========================================

def clean_text(text):
    """
    Chỉ giữ lại chữ cái và số, loại bỏ nhiễu đặc biệt.
    Chuyển đổi các ký tự dễ nhầm lẫn.
    """
    # Thay thế các ký tự đặc biệt thường bị nhận nhầm trước khi xóa
    text = text.replace('|', 'I').replace('l', 'I') 
    
    # Chỉ giữ A-Z và 0-9
    clean = re.sub(r'[^a-zA-Z0-9]', '', text).upper()
    return clean

def sort_contours_grid(cnts, row_sensitivity=15):
    """Sắp xếp contour theo lưới (Trái->Phải, Trên->Dưới)"""
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    c_boxes = list(zip(cnts, boundingBoxes))
    
    # Sắp xếp theo Y trước (để chia dòng)
    c_boxes.sort(key=lambda b: b[1][1]) 

    rows = []
    current_row = []
    last_y = -999

    for c, box in c_boxes:
        y = box[1]
        h = box[3]
        # Nếu box này nằm cùng dòng với box trước (chênh lệch y không quá lớn)
        if y - last_y < row_sensitivity and last_y != -999:
            current_row.append((c, box))
        else:
            if current_row:
                # Dòng cũ đã xong, sắp xếp dòng cũ theo X (Trái -> Phải)
                current_row.sort(key=lambda b: b[1][0])
                rows.extend(current_row)
            # Bắt đầu dòng mới
            current_row = [(c, box)]
            last_y = y
    
    # Thêm dòng cuối cùng
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        rows.extend(current_row)

    return [item[0] for item in rows]

def preprocess_roi_for_ocr(roi):
    """
    Chuẩn bị ảnh cắt (ROI) để OCR tốt nhất:
    1. Phóng to (Upscale).
    2. Chuyển xám & Nhị phân hóa (Threshold).
    3. Thêm viền trắng (Padding).
    """
    # 1. Phóng to ảnh lên 3 lần (Giúp Tesseract đọc chữ nhỏ tốt hơn)
    roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 2. Threshold OTSU để tách chữ đen trên nền trắng
    # Lưu ý: Code trong ảnh là chữ đen nền trắng -> Binary thường (không INV) hoặc INV tùy theo background
    # Ở đây dùng THRESH_BINARY vì text màu đen, nền trắng, sau threshold text sẽ là đen (0) nền trắng (255)
    # Tesseract thích chữ đen nền trắng hoặc ngược lại đều được, nhưng chuẩn nhất là chữ đen nền trắng.
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Thêm viền trắng xung quanh để chữ không bị sát mép
    thresh = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    return thresh

def process_image(image_file):
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Xử lý tìm khung Button
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dùng HSV để bắt màu trắng (nền button)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180]) 
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Morph Open để xóa nhiễu chữ, giữ lại khối button
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) # Tăng kernel lên chút
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    img_h, img_w = img.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Lọc khung code (Hình chữ nhật nằm ngang)
        if area > 1000 and 2.0 < aspect_ratio < 7.0 and w < (img_w * 0.8):
            # Kiểm tra độ sáng trung bình để loại bỏ các box nền tối (như Telegram)
            roi_check = gray[y:y+h, x:x+w]
            mean_val = cv2.mean(roi_check)[0]
            if mean_val > 160: # Nền sáng
                valid_contours.append(c)

    detected_codes = []
    
    if valid_contours:
        # Sắp xếp contour theo lưới
        valid_contours = sort_contours_grid(valid_contours, row_sensitivity=25)
        
        for idx, c in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(c)
            
            # Crop vùng ảnh (Padding vào trong một chút để bỏ viền button)
            pad_y = int(h * 0.15) # Bỏ 15% trên dưới
            pad_x = int(w * 0.05) # Bỏ 5% trái phải
            
            roi = gray[y+pad_y : y+h-pad_y, x+pad_x : x+w-pad_x]
            
            if roi.size == 0: continue
            
            try:
                # --- XỬ LÝ ẢNH TRƯỚC KHI OCR ---
                processed_roi = preprocess_roi_for_ocr(roi)
                
                # Cấu hình Tesseract: 
                # --psm 7: Treat the image as a single text line.
                # Bỏ whitelist cứng để nó đọc tự nhiên, sau đó mình clean bằng Python
                config = r'--psm 7'
                
                text = pytesseract.image_to_string(processed_roi, config=config)
                
                # Lọc sạch text
                final_code = clean_text(text)
                
                if len(final_code) >= 4: # Chỉ lấy nếu độ dài >= 4 ký tự
                    detected_codes.append(final_code)
                    
                    # Vẽ kết quả lên ảnh
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Vẽ nền đen cho chữ dễ đọc
                    cv2.rectangle(img, (x, y+h-25), (x+w, y+h), (0,0,0), -1)
                    cv2.putText(img, final_code, (x + 10, y + h - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error processing box {idx}: {e}")
                continue

    return img, detected_codes, mask_clean

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Tool Scan Code OKVIP", layout="wide")
st.title("🧩 Tool Quét Code - Optimized")
st.markdown("Đã tối ưu: **Upscale ảnh** + **Tắt vẽ lại contour** + **Lọc nhiễu Regex**")

uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    processed_img, codes, debug_mask = process_image(uploaded_file)
    
    with col1:
        st.subheader("Ảnh đã xử lý")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("Kết quả Code")
        if codes:
            # Join with newline
            txt = "\n".join(codes)
            st.text_area("Copy tại đây:", value=txt, height=500)
            st.success(f"Tìm thấy {len(codes)} mã.")
        else:
            st.warning("Không tìm thấy mã nào hoặc ảnh quá mờ.")
