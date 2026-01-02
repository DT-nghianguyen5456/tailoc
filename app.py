import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# ==========================================
# CẤU HÌNH & HÀM PHỤ TRỢ
# ==========================================

def clean_text(text):
    """
    Lọc text: Chỉ giữ lại chữ cái và số, viết hoa toàn bộ.
    """
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned.upper()

def sort_contours_grid(cnts, row_sensitivity=20):
    """
    Sắp xếp contour theo dạng lưới (Grid): Trái->Phải, Trên->Dưới
    """
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    c_boxes = list(zip(cnts, boundingBoxes))
    
    # Sắp xếp theo chiều Y trước
    c_boxes.sort(key=lambda b: b[1][1])

    rows = []
    current_row = []
    last_y = -999

    for c, box in c_boxes:
        y = box[1]
        # Nếu cùng một hàng (chênh lệch Y ít)
        if y - last_y < row_sensitivity and last_y != -999:
            current_row.append((c, box))
        else:
            if current_row:
                # Sắp xếp hàng cũ theo X (Trái -> Phải)
                current_row.sort(key=lambda b: b[1][0])
                rows.extend(current_row)
            current_row = [(c, box)]
            last_y = y
    
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        rows.extend(current_row)

    return [item[0] for item in rows]

def pre_process_char_filter(roi_gray):
    """
    Hàm lọc nhiễu nâng cao:
    Tách từng ký tự trong ô, đo kích thước.
    - Xóa nét quá mảnh (|)
    - Xóa nét quá thấp (~ . , -)
    - Chỉ giữ lại chữ cái (đậm và cao)
    """
    # 1. Phân ngưỡng để tách chữ khỏi nền (Chữ trắng trên nền đen cho findContours)
    # Dùng Adaptive Threshold để xử lý tốt dù ánh sáng không đều
    thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Tìm contours các ký tự vụn vặt
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo một ảnh trắng tinh để vẽ lại các chữ "sạch"
    clean_mask = np.ones(roi_gray.shape, dtype="uint8") * 255 # Nền trắng

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        
        # --- BỘ LỌC HÌNH HỌC (QUAN TRỌNG) ---
        # 1. Loại bỏ nhiễu quá nhỏ (diện tích < 15)
        if w * h < 15: continue
            
        # 2. Loại bỏ dấu chấm, dấu ngã, gạch dưới (Chiều cao < 8px)
        if h < 8: continue
            
        # 3. Loại bỏ gạch đứng '|' (Chiều rộng < 4px HOẶC Tỉ lệ cao/rộng > 5)
        # Chữ 'I' hoặc '1' thường đậm hơn (w >= 4) hoặc tỉ lệ không quá dẹt
        ratio = h / float(w)
        if w < 4 or ratio > 5.0: continue
            
        # Nếu vượt qua các bài test trên, đây là chữ cái -> Vẽ lại màu đen lên nền trắng
        cv2.drawContours(clean_mask, [c], -1, 0, -1)
        
    return clean_mask

def process_image(image_file):
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tìm các ô trắng (Button)
    # Ngưỡng cao (200) để bắt màu trắng
    _, thresh_bg = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Đóng các lỗ hổng nếu có
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_bg = cv2.morphologyEx(thresh_bg, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(thresh_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    img_h, img_w = img.shape[:2]

    # Lọc ô code (Hình chữ nhật)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        # Điều kiện: Diện tích đủ lớn, hình chữ nhật ngang, không phải banner to
        if area > 1000 and 2.0 < aspect_ratio < 7.0 and w < (img_w * 0.9):
            valid_contours.append(c)

    detected_codes = []

    if valid_contours:
        # Sắp xếp thứ tự
        valid_contours = sort_contours_grid(valid_contours, row_sensitivity=img_h//20)
        
        for idx, c in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(c)
            
            # Cắt ảnh (Crop) - Thụt vào 5px để bỏ viền đen của nút
            pad = 5
            if h > 2*pad and w > 2*pad:
                roi = gray[y+pad:y+h-pad, x+pad:x+w-pad]
            else:
                roi = gray[y:y+h, x:x+w]
            
            if roi.size == 0: continue

            # --- BƯỚC XỬ LÝ MỚI: TÁI TẠO ẢNH ---
            # Thay vì đọc ảnh gốc, ta lọc bỏ nhiễu và vẽ lại ảnh mới chỉ chứa chữ
            clean_roi = pre_process_char_filter(roi)
            
            # OCR trên ảnh sạch
            # PSM 7: Xem là một dòng văn bản đơn
            # whitelist: Chỉ cho phép chữ và số (để tránh nhận diện rác)
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(clean_roi, config=config)
            
            final_code = clean_text(text)
            
            # Kiểm tra độ dài hợp lý (ít nhất 3 ký tự)
            if len(final_code) >= 3:
                detected_codes.append(final_code)
                # Vẽ khung để người dùng thấy
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, str(idx + 1), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return img, detected_codes

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Trích xuất Code OKVIP (Advanced)", layout="wide")

st.title("🧩 Tool Quét Code - Phiên Bản Khử Nhiễu")
st.markdown("""
**Cải tiến:** Tự động loại bỏ các ký tự đặc biệt như `|`, `~`, `.`, `_` bằng thuật toán phân tích hình học trước khi đọc.
""")

uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🖼️ Ảnh đã xử lý")
        processed_img, codes = process_image(uploaded_file)
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader(f"📋 Kết quả ({len(codes)} mã)")
        if codes:
            with st.container(height=600):
                for idx, code in enumerate(codes):
                    st.text_input(f"Code #{idx+1}", value=code, key=f"code_{idx}")
        else:
            st.warning("Không tìm thấy mã nào hợp lệ.")
