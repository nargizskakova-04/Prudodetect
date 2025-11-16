from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
import pdf2image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Конфигурация
MODEL_PATH = os.getenv('MODEL_PATH', 'best.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))
MAX_IMAGE_SIZE = 4096  # Максимальный размер изображения

# Загружаем модель при старте сервера
print(f"🔄 Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Маппинг классов (адаптируйте под вашу модель)
CLASS_NAMES = {
    0: 'qr',
    1: 'signature', 
    2: 'stamp'
}

def convert_pdf_to_images(pdf_bytes):
    """Конвертирует PDF в список изображений"""
    try:
        # Сохраняем PDF во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        # Конвертируем PDF в изображения (первая страница для упрощения)
        images = pdf2image.convert_from_path(
            tmp_path, 
            dpi=300,  # Высокое качество для детекции
            first_page=1,
            last_page=1  # Только первая страница
        )
        
        # Удаляем временный файл
        os.unlink(tmp_path)
        
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None

def preprocess_image(image):
    """Предобработка изображения"""
    # Конвертируем PIL Image в numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Конвертируем в RGB если нужно
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Ресайзим если изображение слишком большое
    h, w = image.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"📐 Image resized from {w}x{h} to {new_w}x{new_h}")
    
    return image

def run_inference(image_np):
    """Запускает inference на изображении"""
    try:
        # Предобработка
        image_np = preprocess_image(image_np)
        
        # Inference
        results = model(image_np, conf=CONFIDENCE_THRESHOLD)
        
        # Парсим результаты
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Координаты bbox
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Уверенность и класс
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Формируем объект детекции
                detection = {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'class': CLASS_NAMES.get(cls, f'class_{cls}'),
                    'confidence': round(conf, 3)
                }
                
                detections.append(detection)
        
        return detections
    
    except Exception as e:
        print(f"Inference error: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Основной endpoint для детекции"""
    try:
        # Проверяем наличие файла
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # Читаем содержимое файла
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        file_ext = filename.lower().split('.')[-1]
        
        print(f"📄 Processing file: {filename} ({len(file_bytes)} bytes)")
        
        # Обработка в зависимости от типа файла
        if file_ext == 'pdf':
            # PDF → изображения
            images = convert_pdf_to_images(file_bytes)
            
            if images is None or len(images) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Failed to convert PDF to images'
                }), 400
            
            # Берем первую страницу
            image_np = np.array(images[0])
            print(f"✅ PDF converted to image: {image_np.shape}")
        
        else:
            # Обычное изображение (PNG, JPG, JPEG)
            try:
                image = Image.open(io.BytesIO(file_bytes))
                image_np = np.array(image)
                print(f"✅ Image loaded: {image_np.shape}")
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to read image: {str(e)}'
                }), 400
        
        # Запускаем inference
        detections = run_inference(image_np)
        
        print(f"🎯 Found {len(detections)} objects")
        
        # Группируем по классам для статистики
        stats = {}
        for det in detections:
            class_name = det['class']
            stats[class_name] = stats.get(class_name, 0) + 1
        
        return jsonify({
            'success': True,
            'detections': detections,
            'total_count': len(detections),
            'statistics': stats,
            'message': f'Found {len(detections)} objects'
        })
    
    except Exception as e:
        print(f"❌ Error in predict endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Информация о модели"""
    try:
        return jsonify({
            'model_path': MODEL_PATH,
            'classes': CLASS_NAMES,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'model_type': str(type(model).__name__)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"""
    🚀 ML Service starting...
    📊 Port: {port}
    🤖 Model: {MODEL_PATH}
    🎯 Confidence threshold: {CONFIDENCE_THRESHOLD}
    🔧 Debug mode: {debug}
    """)
    
    app.run(
        host='0.0.0.0', 
        port=port,
        debug=debug
    )