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
    
    # Phân ngưỡng (Threshold) để tách phần màu trắng
    # Các ô màu trắng sẽ có giá trị cao (gần 255)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Tìm các đường viền (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_codes = []
    valid_contours = []

    # Lọc các contour để tìm đúng ô chứa mã
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Điều kiện lọc: Diện tích phải đủ lớn và hình dáng chữ nhật ngang
        # Bạn có thể điều chỉnh số 2000 tùy theo độ phân giải ảnh
        if area > 2000 and aspect_ratio > 2.0:
            valid_contours.append(c)

    # Sắp xếp contour từ trên xuống dưới để đọc đúng thứ tự
    if valid_contours:
        # Sắp xếp sơ bộ từ trên xuống dưới
        (valid_contours, _) = sort_contours(valid_contours, method="top-to-bottom")
        
        # Xử lý gom nhóm từng hàng (để sắp xếp trái sang phải trong cùng 1 hàng)
        sorted_final = []
        # Giả sử mỗi hàng cao khoảng h pixels, ta gom nhóm các contour có y gần nhau
        # (Đây là logic đơn giản hóa, với lưới đều nhau thì ổn)
        # Để đơn giản cho demo, ta dùng logic sắp xếp theo tọa độ Y trước, 
        # sau đó gom nhóm các box có Y gần nhau để sort theo X.
        
        # NOTE: Với lưới Grid rõ ràng như ảnh, ta có thể dùng thư viện imutils hoặc logic custom.
        # Ở đây mình dùng logic đọc tuần tự theo bounding box đã sort top-to-bottom.
        # Để chính xác tuyệt đối trái-phải, cần gom nhóm theo hàng (row).
        
        # Logic đơn giản: Cắt từng ô và nhận diện
        for c in valid_contours:
            x, y, w, h = cv2.boundingRect(c)
            
            # Cắt ảnh (Crop) vùng ô trắng (thêm margin nhỏ để tránh mất nét)
            roi = img[y+5:y+h-5, x+5:x+w-5] 
            
            # Xử lý ảnh con để OCR tốt hơn
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Dùng Tesseract để đọc
            # config='--psm 6' phù hợp cho khối văn bản đơn dòng
            text = pytesseract.image_to_string(roi_thresh, config='--psm 6')
            
            cleaned = clean_text(text)
            
            if cleaned: # Chỉ thêm nếu đọc được chữ
                detected_codes.append(cleaned)
                
                # Vẽ hình chữ nhật lên ảnh gốc để visualize (tùy chọn)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, cleaned, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
