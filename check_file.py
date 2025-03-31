import os

image_path = "static/output/images.png"

if os.path.exists(image_path):
    print("✅ Ảnh tồn tại.")
else:
    print("❌ Ảnh không tồn tại, kiểm tra lại đường dẫn!")
