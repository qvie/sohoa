from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import psycopg2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import uuid
import time
from datetime import datetime 
from paddleocr import PaddleOCR
import base64
from fpdf import FPDF
import hashlib
from Crypto.Util.Padding import unpad, pad
from Crypto.Cipher import AES
import random
import string
from flask import send_from_directory
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_FOLDER = 'static/output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Kết nối PostgreSQL
DATABASE_URL = "dbname='ocr_dbs' user='postgres' password='123456' host='localhost' port='5432'"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

SECRET_KEY = "my_secret_key_123"  # 🔐 Đổi thành key an toàn hơn
BLOCK_SIZE = 16  # AES block size

# Khởi tạo PaddleOCR (chỉ khởi tạo 1 lần)
ocr = PaddleOCR(lang="vi")

def get_key():
    """Tạo khóa AES từ SECRET_KEY (dùng SHA-256 để tạo key 32 bytes)."""
    return hashlib.sha256(SECRET_KEY.encode()).digest()

def encrypt_text(text):
    """Mã hóa văn bản với AES-256."""
    key = get_key()
    cipher = AES.new(key, AES.MODE_ECB)
    padded_data = pad(text.encode(), AES.block_size)
    encrypted = cipher.encrypt(padded_data)
    return base64.b64encode(encrypted).decode()

def decrypt_text(encrypted_text):
    """Giải mã văn bản AES-256."""
    key = get_key()
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted_padded = cipher.decrypt(base64.b64decode(encrypted_text))
    return unpad(decrypted_padded, AES.block_size).decode()

def preprocess_image(image_path):
    """Tiền xử lý ảnh trước khi đưa vào OCR"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cân bằng histogram
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Loại bỏ nhiễu
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return denoised

def generate_passcode():
    """Tạo passcode ngẫu nhiên 8 ký tự."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def process_image(image_path):
    """Nhận diện văn bản từ hình ảnh bằng PaddleOCR"""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return "Không thể đọc ảnh", "", "", 0  # Trả về 0% nếu lỗi

    # Lưu ảnh đã xử lý để đưa vào OCR
    temp_processed_path = os.path.join(OUTPUT_FOLDER, "temp_processed.png")
    cv2.imwrite(temp_processed_path, processed_image)

    # Nhận diện văn bản với PaddleOCR
    result = ocr.ocr(temp_processed_path, cls=True)
    
    extracted_text = ""
    confidence_scores = []
    
    for line in result:
        if line:
            for word_info in line:
                text, confidence = word_info[1]
                extracted_text += text + " "
                confidence_scores.append(confidence)

    # Tính độ chính xác trung bình
    average_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    # Lưu ảnh kết quả
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())
    output_image_path = os.path.join(OUTPUT_FOLDER, f'ocr_result_{unique_id}_{timestamp}.png')

    return extracted_text.strip(), image_path, output_image_path, round(average_confidence * 100, 2)  # Đổi sang %

def save_document(name, file_path, ocr_content, category="Uncategorized", uploaded_by=None):
    upload_datetime = datetime.now()
    uploaded_by = uploaded_by if uploaded_by else "Unknown"
    passcode = generate_passcode()  # Tạo passcode cho tài liệu
    encrypted_content = encrypt_text(ocr_content)  # 🔐 Mã hóa nội dung OCR

    cursor.execute("""
        INSERT INTO documents (name, file_path, ocr_content, category, uploaded_by, upload_date, passcode, indexed)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
    """, (name, file_path, encrypted_content, category, uploaded_by, upload_datetime, passcode, False))
    
    doc_id = cursor.fetchone()[0]
    conn.commit()
    return doc_id, passcode  # Trả về cả ID và passcode

def text_to_image(text, output_path):
    font = ImageFont.truetype("arial.ttf", 36)  # Font chữ
    max_width = 800  # Chiều rộng tối đa của ảnh
    margin = 50  # Lề trái, phải

    # Chia văn bản thành các dòng phù hợp với chiều rộng ảnh
    lines = []
    words = text.split()
    current_line = ""

    image = Image.new("RGB", (max_width, 400), color=(255, 255, 255))  # Ảnh tạm thời
    draw = ImageDraw.Draw(image)

    for word in words:
        test_line = current_line + " " + word if current_line else word
        test_size = draw.textbbox((0, 0), test_line, font=font)[2]  # Lấy chiều rộng

        if test_size < max_width - 2 * margin:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)  # Thêm dòng cuối cùng

    # Tính toán chiều cao cần thiết cho ảnh
    line_height = draw.textbbox((0, 0), "A", font=font)[3] - draw.textbbox((0, 0), "A", font=font)[1]  # Chiều cao một dòng
    image_height = max(400, len(lines) * (line_height + 10) + 100)  # Tăng chiều cao nếu cần

    # Tạo ảnh mới với chiều cao chính xác
    image = Image.new("RGB", (max_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Vẽ từng dòng lên ảnh
    y_position = 50
    for line in lines:
        draw.text((margin, y_position), line, font=font, fill=(0, 0, 0))
        y_position += line_height + 10

    image.save(output_path)

def save_metadata(document_id, metadata):
    for key, value in metadata.items():
        cursor.execute("""
            INSERT INTO metadata (document_id, key, value)
            VALUES (%s, %s, %s)
        """, (document_id, key, value))
    conn.commit()  # ✅ Thêm commit để lưu vào DB

def save_tags(document_id, tags):
    for tag in tags:
        cursor.execute("""
            INSERT INTO tags (document_id, tag)
            VALUES (%s, %s)
        """, (document_id, tag))
    conn.commit()  # ✅ Thêm commit để lưu vào DB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    extracted_text, original_image_path, output_image_path, accuracy = process_image(file_path)

    category = request.form.get('category', 'Uncategorized')
    uploaded_by = request.form.get('uploaded_by', 'Unknown')

    doc_id = save_document(file.filename, file_path, extracted_text, category=category, uploaded_by=uploaded_by)

    return render_template('result.html', data={
        'content': extracted_text,
        'original_image': f'output/{os.path.basename(original_image_path)}',
        'ocr_image': f'output/{os.path.basename(output_image_path)}',
        'uploaded_by': uploaded_by,
        'accuracy': f"{accuracy:.2f}%" if accuracy is not None else "N/A"  # 👈 Hiển thị accuracy nếu có
    })
    
@app.route('/text-to-image', methods=['POST'])
def generate_image():
    text = request.form.get('text')  # Nhận nội dung văn bản từ form
    output_path = os.path.join(OUTPUT_FOLDER, 'text_image.png')  # Đường dẫn ảnh đầu ra
    text_to_image(text, output_path)  # Gọi hàm để chuyển văn bản thành ảnh

    return send_file(output_path, as_attachment=True, download_name="generated_image.png")
@app.route('/download_image/<filename>', methods=['GET'])
def download_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/search', methods=['GET'])
def search_documents():
    query = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()
    uploaded_by = request.args.get('uploaded_by', '').strip()
    passcode = request.args.get('passcode', '').strip()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()

    print(f"🔍 Debug: Query nhập vào - {query}")
    print(f"🔍 Debug: Danh mục - {category}, Người tải lên - {uploaded_by}, Từ ngày - {start_date}, Đến ngày - {end_date}")

    sql = """
        SELECT id, name, category, ocr_content, uploaded_by, upload_datetime 
        FROM documents WHERE 1=1
    """
    params = []

    if passcode:
        sql += " AND passcode = %s"
        params.append(passcode)

    if query:
        sql += " AND (LOWER(ocr_content) LIKE %s OR LOWER(category) LIKE %s OR LOWER(name) LIKE %s)"
        params.extend(['%' + query.lower() + '%'] * 3)

    if category:
        sql += " AND LOWER(category) = %s"
        params.append(category.lower())

    if uploaded_by:
        sql += " AND LOWER(uploaded_by) = %s"
        params.append(uploaded_by.lower())

    if start_date and end_date:
        sql += " AND upload_datetime BETWEEN %s AND %s"
        params.extend([start_date, end_date])

    sql += " ORDER BY upload_datetime DESC"  # Sắp xếp theo ngày tải lên mới nhất

    with conn.cursor() as cursor:
        cursor.execute(sql, tuple(params))
        documents = cursor.fetchall()

    return render_template('search.html', documents=documents, query=query, passcode=passcode)

@app.route('/export-pdf', methods=['GET', 'POST'])
def export_pdf():
    if request.method == 'GET':
        return "Vui lòng gửi yêu cầu POST với dữ liệu văn bản để tạo PDF."

    text = request.form.get('text', 'Không có dữ liệu')
    pdf_path = os.path.join(OUTPUT_FOLDER, "ocr_result.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Chia văn bản thành nhiều dòng nếu quá dài
    lines = text.split("\n")
    for line in lines:
        pdf.multi_cell(0, 10, txt=line, align='L')

    pdf.output(pdf_path)
    
    return send_file(pdf_path, as_attachment=True, download_name="ocr_result.pdf")
@app.route('/get_image/<path:filename>')
def get_image(filename):
    return send_from_directory('static/output', filename)

@app.route('/document/<int:doc_id>')
def view_document(doc_id):
    cursor.execute("SELECT name, file_path, ocr_content, category FROM documents WHERE id = %s", (doc_id,))
    doc = cursor.fetchone()

    if not doc:
        return "Tài liệu không tồn tại!", 404

    decrypted_text = decrypt_text(doc[2])  # 🔓 Giải mã nội dung OCR trước khi hiển thị

    return render_template('document.html', document={
        'filename': doc[0],
        'file_path': doc[1].replace("static/", ""),
        'ocr_text': decrypted_text,  # 🔓 Hiển thị văn bản gốc đã giải mã
        'category': doc[3]
    })
if __name__ == '__main__':
    app.run(debug=True, port=5001)
