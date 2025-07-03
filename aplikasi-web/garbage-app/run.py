from app import app, load_model
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("🚀 Memulai Garbage Detection API...")

    if load_model():
        print("✅ Model berhasil dimuat.")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Gagal memuat model.")