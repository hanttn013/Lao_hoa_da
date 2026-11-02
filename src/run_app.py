import os
import sys
import webbrowser
import threading
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from web_app.server import app, load_all_models

def open_browser(port):
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    print(f"Opening web browser at: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print("Could not auto-open browser:", e)

def main():
    port = int(os.environ.get("PORT", 5087))
    print(f"=== Starting Facial Wrinkle Assessment Platform on port {port} ===")
    load_all_models()
    
    # Launch browser thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    main()
