"""
main.py
Main entry point for the Multi-Modal Biometric Attendance & Security System.
Run:
    python main.py        -> Launches the interactive Desktop GUI dashboard.
    python main.py --web  -> Launches the Web Server on http://localhost:5000.
    python main.py --cli  -> Runs headless multi-modal verification in terminal.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import argparse
from core.database import init_db

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Biometric Attendance System")
    parser.add_argument("--cli", action="store_true", help="Run in terminal CLI mode without GUI")
    parser.add_argument("--web", action="store_true", help="Run in Web browser mode at http://localhost:5000")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server (default: 5000)")
    args = parser.parse_args()

    # Ensure SQLite database is initialized
    init_db()

    if args.web:
        import webbrowser
        import threading
        print("=" * 60)
        print(f"🚀 Starting Biometric Attendance Web Portal on localhost:{args.port}...")
        print(f"👉 Opening browser: http://localhost:{args.port}")
        print("=" * 60)
        # Auto-open browser after 1.5 seconds
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
        from app_web import app
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    elif args.cli:
        print("Starting in Headless/CLI mode...")
        import secure_verify
        secure_verify.main()
    else:
        print("Launching Desktop GUI Dashboard...")
        import app_gui
        app = app_gui.BiometricAttendanceApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()

if __name__ == "__main__":
    main()
