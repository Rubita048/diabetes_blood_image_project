import os
from datetime import datetime

def log_message(message, log_path="results/logs/training_log.txt"):
    """Log dosyasına zaman damgalı mesaj ekler."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(f"📝 Log: {message}")
