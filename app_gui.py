"""
app_gui.py
Modern Desktop GUI for Multi-Modal Biometric Attendance System using CustomTkinter.
Features:
- Real-time video processing with MediaPipe Blink Liveness Detection
- Deep learning face recognition
- Integrated voice verification with audio challenge
- Full enrollment suite (Face + Voice)
- Real-time SQLite database viewer and CSV export
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import time
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np

# Core modules
from core.database import (
    init_db, add_or_get_user, save_face_embedding, save_voice_embedding,
    get_all_face_embeddings, get_voice_embedding, get_all_users,
    record_attendance, get_attendance_logs, export_attendance_to_csv
)
from core.liveness import LivenessDetector
from core.face_engine import FaceEngine
from core.voice_engine import VoiceEngine, ChallengeSession
from core.migration import migrate_legacy_data
from core.verification_service import process_voice_challenge, finalize_attendance
from config import (
    CAMERA_INDEX, FACE_THRESHOLD, VOICE_THRESHOLD,
    EAR_THRESHOLD, REQUIRED_BLINKS, CODE_LENGTH,
    MAX_CODE_ATTEMPTS, CODE_EXPIRY_SECONDS
)

# Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class BiometricAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Biometric Attendance & Security System")
        self.geometry("1180x760")
        self.minsize(1050, 700)

        # Initialize engines
        self.liveness_detector = LivenessDetector(ear_threshold=EAR_THRESHOLD, required_blinks=REQUIRED_BLINKS)
        self.face_engine = FaceEngine(threshold=FACE_THRESHOLD)
        self.voice_engine = VoiceEngine(threshold=VOICE_THRESHOLD)

        # App state
        self.cap = None
        self.is_camera_running = False
        self.camera_index = CAMERA_INDEX
        self.verification_step = 1  # 1: Liveness, 2: Face, 3: Voice, 4: Done
        self.verified_user = None
        self.verified_face_conf = 0.0
        self.enrolled_faces = {}
        self.is_recording_voice = False
        self.enroll_face_samples = []

        # Load enrolled database
        self.reload_enrolled_data()

        # Layout UI
        self.setup_ui()

        # Start clock update
        self.update_clock()

    def reload_enrolled_data(self):
        self.enrolled_faces = get_all_face_embeddings()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ----------------- SIDEBAR -----------------
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="⚡ BIO-SECURE",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar,
            text="Multi-Modal Attendance",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Nav Buttons
        self.btn_nav_verify = ctk.CTkButton(
            self.sidebar, text="📸 Live Verification",
            command=lambda: self.show_page("verify"),
            height=40, font=ctk.CTkFont(weight="bold")
        )
        self.btn_nav_verify.grid(row=2, column=0, padx=15, pady=8, sticky="ew")

        self.btn_nav_enroll = ctk.CTkButton(
            self.sidebar, text="👤 User Enrollment",
            command=lambda: self.show_page("enroll"),
            height=40
        )
        self.btn_nav_enroll.grid(row=3, column=0, padx=15, pady=8, sticky="ew")

        self.btn_nav_logs = ctk.CTkButton(
            self.sidebar, text="📋 Attendance Records",
            command=lambda: self.show_page("logs"),
            height=40
        )
        self.btn_nav_logs.grid(row=4, column=0, padx=15, pady=8, sticky="ew")

        self.btn_nav_settings = ctk.CTkButton(
            self.sidebar, text="⚙ System Settings",
            command=lambda: self.show_page("settings"),
            height=40
        )
        self.btn_nav_settings.grid(row=5, column=0, padx=15, pady=8, sticky="ew")

        # Sidebar Footer
        self.lbl_clock = ctk.CTkLabel(
            self.sidebar, text="", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_clock.grid(row=7, column=0, padx=15, pady=(5, 15))

        # ----------------- MAIN PAGES CONTAINER -----------------
        self.pages_container = ctk.CTkFrame(self, corner_radius=10)
        self.pages_container.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.pages_container.grid_columnconfigure(0, weight=1)
        self.pages_container.grid_rowconfigure(0, weight=1)

        self.pages = {}
        self.pages["verify"] = self.build_verify_page()
        self.pages["enroll"] = self.build_enroll_page()
        self.pages["logs"] = self.build_logs_page()
        self.pages["settings"] = self.build_settings_page()

        self.show_page("verify")

    def show_page(self, page_name):
        for name, frame in self.pages.items():
            if name == page_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

        # Update sidebar button colors
        buttons = {
            "verify": self.btn_nav_verify,
            "enroll": self.btn_nav_enroll,
            "logs": self.btn_nav_logs,
            "settings": self.btn_nav_settings
        }
        for name, btn in buttons.items():
            btn.configure(fg_color="#1f538d" if name == page_name else "#2b2b2b")

        if page_name == "logs":
            self.refresh_logs_table()

    def update_clock(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.lbl_clock.configure(text=now)
        self.after(1000, self.update_clock)

    # ==============================================================
    # PAGE 1: VERIFICATION & ATTENDANCE DASHBOARD
    # ==============================================================
    def build_verify_page(self):
        page = ctk.CTkFrame(self.pages_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=3)
        page.grid_columnconfigure(1, weight=2)
        page.grid_rowconfigure(0, weight=1)

        # Left Column: Video Feed
        left_frame = ctk.CTkFrame(page)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        header_lbl = ctk.CTkLabel(left_frame, text="Real-Time Camera Monitor", font=ctk.CTkFont(size=16, weight="bold"))
        header_lbl.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.video_canvas = ctk.CTkLabel(left_frame, text="Camera Offline", fg_color="#111", corner_radius=8)
        self.video_canvas.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        btn_row = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.btn_toggle_cam = ctk.CTkButton(
            btn_row, text="▶ Start Camera", fg_color="#2e7d32", hover_color="#1b5e20",
            command=self.toggle_camera
        )
        self.btn_toggle_cam.pack(side="left", padx=5)

        self.btn_reset_verify = ctk.CTkButton(
            btn_row, text="🔄 Reset Pipeline", fg_color="#455a64", hover_color="#37474f",
            command=self.reset_verification_pipeline
        )
        self.btn_reset_verify.pack(side="left", padx=5)

        # Right Column: Step-by-Step Security Pipeline
        right_frame = ctk.CTkFrame(page)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)

        steps_title = ctk.CTkLabel(right_frame, text="Authentication Status", font=ctk.CTkFont(size=16, weight="bold"))
        steps_title.pack(padx=15, pady=(15, 10), anchor="w")

        # Step 1 Card: Liveness
        self.card_liveness = ctk.CTkFrame(right_frame)
        self.card_liveness.pack(fill="x", padx=15, pady=6)
        self.lbl_liveness_title = ctk.CTkLabel(self.card_liveness, text="Step 1: Liveness (Blink Detection)", font=ctk.CTkFont(weight="bold"))
        self.lbl_liveness_title.pack(anchor="w", padx=10, pady=(5, 0))
        self.lbl_liveness_status = ctk.CTkLabel(self.card_liveness, text="⏳ Waiting for face...", text_color="gray")
        self.lbl_liveness_status.pack(anchor="w", padx=10, pady=(0, 5))

        # Step 2 Card: Face Recognition
        self.card_face = ctk.CTkFrame(right_frame)
        self.card_face.pack(fill="x", padx=15, pady=6)
        self.lbl_face_title = ctk.CTkLabel(self.card_face, text="Step 2: Face Recognition", font=ctk.CTkFont(weight="bold"))
        self.lbl_face_title.pack(anchor="w", padx=10, pady=(5, 0))
        self.lbl_face_status = ctk.CTkLabel(self.card_face, text="⏸ Locked (Pass Step 1 first)", text_color="gray")
        self.lbl_face_status.pack(anchor="w", padx=10, pady=(0, 5))

        # Step 3 Card: Voice Verification
        self.card_voice = ctk.CTkFrame(right_frame)
        self.card_voice.pack(fill="x", padx=15, pady=6)
        self.lbl_voice_title = ctk.CTkLabel(self.card_voice, text="Step 3: Voice Verification", font=ctk.CTkFont(weight="bold"))
        self.lbl_voice_title.pack(anchor="w", padx=10, pady=(5, 0))
        self.lbl_voice_status = ctk.CTkLabel(self.card_voice, text="⏸ Locked (Pass Step 2 first)", text_color="gray")
        self.lbl_voice_status.pack(anchor="w", padx=10, pady=(0, 2))

        self.lbl_voice_prompt = ctk.CTkLabel(self.card_voice, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color="#ffb300")
        self.lbl_voice_prompt.pack(anchor="w", padx=10, pady=2)

        self.btn_speak = ctk.CTkButton(
            self.card_voice, text="🎙 Speak & Verify", state="disabled",
            command=self.trigger_voice_verification, height=32
        )
        self.btn_speak.pack(fill="x", padx=10, pady=(5, 10))

        # Final Result Banner
        self.result_banner = ctk.CTkLabel(
            right_frame, text="System Ready", font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#333", corner_radius=8, height=45
        )
        self.result_banner.pack(fill="x", padx=15, pady=(15, 10))

        return page

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.is_camera_running:
            return
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot access camera index {self.camera_index}.")
            return
        self.is_camera_running = True
        self.btn_toggle_cam.configure(text="⏹ Stop Camera", fg_color="#c62828", hover_color="#8e0000")
        self.liveness_detector.reset()
        self.update_video_frame()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_toggle_cam.configure(text="▶ Start Camera", fg_color="#2e7d32", hover_color="#1b5e20")
        self.video_canvas.configure(image="", text="Camera Offline")

    def reset_verification_pipeline(self):
        self.verification_step = 1
        self.verified_user = None
        self.verified_face_conf = 0.0
        self.liveness_detector.reset()

        self.lbl_liveness_status.configure(text="⏳ Waiting for face blink...", text_color="yellow")
        self.lbl_face_status.configure(text="⏸ Locked (Pass Step 1 first)", text_color="gray")
        self.lbl_voice_status.configure(text="⏸ Locked (Pass Step 2 first)", text_color="gray")
        self.lbl_voice_prompt.configure(text="")
        self.btn_speak.configure(state="disabled")
        self.result_banner.configure(text="System Ready", fg_color="#333")

    def update_video_frame(self):
        if not self.is_camera_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # Mirror preview
            display_frame = frame.copy()

            # STEP 1: Liveness detection
            if self.verification_step == 1:
                passed, ear, blinks, ann_frame = self.liveness_detector.process_frame(frame)
                display_frame = ann_frame
                self.lbl_liveness_status.configure(
                    text=f"Blinks: {blinks}/1  (EAR: {ear:.2f})",
                    text_color="#00e676" if passed else "#ffb300"
                )
                if passed:
                    self.verification_step = 2
                    self.lbl_liveness_status.configure(text="✔ Passed! Natural blink detected.", text_color="#00e676")
                    self.lbl_face_status.configure(text="🔍 Scanning face...", text_color="yellow")

            # STEP 2: Face Recognition
            elif self.verification_step == 2:
                name, sim, bbox, ann_frame = self.face_engine.process_frame(frame, self.enrolled_faces)
                display_frame = ann_frame
                if name and name != "Unknown":
                    self.verified_user = name
                    self.verified_face_conf = sim
                    self.verification_step = 3
                    self.lbl_face_status.configure(
                        text=f"✔ Identified: {name} ({sim*100:.1f}%)",
                        text_color="#00e676"
                    )
                    # Initialize single-use random 4-digit challenge session
                    self.challenge_session = ChallengeSession(length=CODE_LENGTH)
                    code_formatted = " - ".join(list(self.challenge_session.code))
                    self.lbl_voice_status.configure(
                        text=f"Speak 4-digit code (Attempt 1/{MAX_CODE_ATTEMPTS}):",
                        text_color="#ffb300"
                    )
                    self.lbl_voice_prompt.configure(text=f"Speak:  \"{code_formatted}\"")
                    self.btn_speak.configure(state="normal")
                elif name == "Unknown":
                    self.lbl_face_status.configure(text="⚠ Face not recognized in database", text_color="#ff5252")

            # Convert to RGB and display on Tkinter canvas
            rgb_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_img.shape[:2]
            target_w = 640
            target_h = int(h * (target_w / w))
            rgb_img = cv2.resize(rgb_img, (target_w, target_h))

            pil_img = Image.fromarray(rgb_img)
            imgtk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            self.video_canvas.configure(image=imgtk, text="")

        self.after(25, self.update_video_frame)

    def trigger_voice_verification(self):
        if self.is_recording_voice or not self.verified_user:
            return

        self.is_recording_voice = True
        self.btn_speak.configure(state="disabled", text="🎤 Recording (3s)...")

        def record_thread():
            try:
                import random
                audio = self.voice_engine.record_audio(duration=3.0)
                if not hasattr(self, 'challenge_session') or self.challenge_session is None:
                    self.challenge_session = ChallengeSession(length=CODE_LENGTH)

                self.challenge_session.record_attempt()
                res = process_voice_challenge(
                    user_name=self.verified_user,
                    face_conf=self.verified_face_conf,
                    audio=audio,
                    expected_code=self.challenge_session.code,
                    method="Desktop GUI (FaceNet + Spoken Code + Liveness)"
                )

                def on_done():
                    self.is_recording_voice = False
                    self.btn_speak.configure(text="🎙 Speak & Verify")
                    if res.success:
                        heard_fmt = " ".join(list(res.heard_digits)) if res.heard_digits else ""
                        self.lbl_voice_status.configure(
                            text=f"✔ Code Verified! Heard: {heard_fmt}",
                            text_color="#00e676"
                        )
                        self.result_banner.configure(
                            text=f"✅ {res.message}", fg_color="#1b5e20"
                        )
                        self.verification_step = 4
                    else:
                        if not self.challenge_session.has_attempts_left() or self.challenge_session.is_expired():
                            self.lbl_voice_status.configure(
                                text=f"❌ {res.message}. Max attempts reached. Restarting.",
                                text_color="#ff5252"
                            )
                            self.verification_step = 1
                            self.liveness_detector.reset()
                        else:
                            # Generate new random code for next attempt
                            self.challenge_session.code = "".join([str(random.randint(0, 9)) for _ in range(CODE_LENGTH)])
                            new_code_fmt = " - ".join(list(self.challenge_session.code))
                            self.lbl_voice_prompt.configure(text=f"Speak:  \"{new_code_fmt}\"")
                            self.lbl_voice_status.configure(
                                text=f"⚠ {res.message}. New code issued (Attempt {self.challenge_session.attempts+1}/{MAX_CODE_ATTEMPTS}).",
                                text_color="#ffb300"
                            )
                            self.btn_speak.configure(state="normal")

                self.after(0, on_done)
            except Exception as e:
                self.is_recording_voice = False
                self.after(0, lambda err=e: messagebox.showerror("Audio Error", str(err)))
                self.after(0, lambda: self.btn_speak.configure(state="normal", text="🎙 Speak & Verify"))

        threading.Thread(target=record_thread, daemon=True).start()

    # ==============================================================
    # PAGE 2: USER ENROLLMENT
    # ==============================================================
    def build_enroll_page(self):
        page = ctk.CTkFrame(self.pages_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(page, text="Enroll New User", font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(15, 10), sticky="w")

        content = ctk.CTkFrame(page)
        content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        # Form Inputs (Left)
        left = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        lbl_name = ctk.CTkLabel(left, text="Full Name / User ID:", font=ctk.CTkFont(weight="bold"))
        lbl_name.pack(anchor="w", pady=(0, 5))

        self.entry_enroll_name = ctk.CTkEntry(left, placeholder_text="e.g. STUDENT_NAME", height=40, font=ctk.CTkFont(size=14))
        self.entry_enroll_name.pack(fill="x", pady=(0, 20))

        # Face enrollment card (Face-Only enrollment)
        f_card = ctk.CTkFrame(left)
        f_card.pack(fill="x", pady=10)
        ctk.CTkLabel(f_card, text="Face Profile Capture (5 Samples)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
        self.lbl_face_enroll_status = ctk.CTkLabel(f_card, text="0 / 5 Face samples captured", text_color="gray")
        self.lbl_face_enroll_status.pack(anchor="w", padx=15, pady=(0, 10))

        self.btn_capture_face = ctk.CTkButton(
            f_card, text="📸 Capture Face Samples", command=self.capture_face_samples_interactive, height=35
        )
        self.btn_capture_face.pack(fill="x", padx=15, pady=(0, 15))

        # Enrolled Users List (Right)
        right = ctk.CTkFrame(content)
        right.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(right, text="Registered Users in Database", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.users_listbox = tk.Listbox(
            right, bg="#2b2b2b", fg="white", font=("Arial", 11), selectbackground="#1f538d", borderwidth=0, highlightthickness=0
        )
        self.users_listbox.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")

        self.btn_refresh_users = ctk.CTkButton(
            right, text="🔄 Refresh User List", command=self.refresh_users_list, height=35
        )
        self.btn_refresh_users.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="ew")

        self.refresh_users_list()
        return page

    def capture_face_samples_interactive(self):
        name = self.entry_enroll_name.get().strip().upper().replace(" ", "_")
        if not name:
            messagebox.showwarning("Input Required", "Please enter a user name first.")
            return

        self.btn_capture_face.configure(state="disabled", text="Capturing from webcam...")
        
        def run_capture():
            cam = cv2.VideoCapture(self.camera_index)
            if not cam.isOpened():
                self.after(0, lambda: messagebox.showerror("Error", "Cannot open camera."))
                self.after(0, lambda: self.btn_capture_face.configure(state="normal", text="📸 Capture Face Samples"))
                return

            captured = 0
            while captured < 5:
                ret, frame = cam.read()
                if not ret:
                    break
                bbox = self.face_engine.detect_face_bbox(frame)
                if bbox is not None:
                    x, y, w, h = bbox
                    face_roi = frame[y:y+h, x:x+w]
                    emb = self.face_engine.get_embedding(face_roi)
                    if emb is not None:
                        save_face_embedding(name, emb)
                        captured += 1
                        self.after(0, lambda c=captured: self.lbl_face_enroll_status.configure(
                            text=f"✔ Captured {c} / 5 samples", text_color="#00e676"
                        ))
                        time.sleep(0.4)
                time.sleep(0.1)

            cam.release()
            self.reload_enrolled_data()
            self.after(0, lambda: self.btn_capture_face.configure(state="normal", text="📸 Capture Face Samples"))
            self.after(0, lambda: messagebox.showinfo("Enrollment", f"Successfully saved 5 face embeddings for {name}!"))
            self.after(0, self.refresh_users_list)

        threading.Thread(target=run_capture, daemon=True).start()

    def record_voice_samples_interactive(self):
        name = self.entry_enroll_name.get().strip().upper().replace(" ", "_")
        if not name:
            messagebox.showwarning("Input Required", "Please enter a user name first.")
            return

        self.btn_record_voice_enroll.configure(state="disabled", text="🎤 Recording (3 samples)...")

        def run_voice():
            embeddings = []
            for i in range(3):
                self.after(0, lambda idx=i: self.lbl_voice_enroll_status.configure(
                    text=f"Recording sample {idx+1}/3... Speak now!", text_color="#ffb300"
                ))
                audio = self.voice_engine.record_audio(duration=3.0)
                emb = self.voice_engine.extract_mfcc_embedding(audio)
                if emb is not None:
                    embeddings.append(emb)
                time.sleep(0.5)

            if len(embeddings) > 0:
                avg_emb = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    avg_emb = avg_emb / norm
                save_voice_embedding(name, avg_emb)
                self.after(0, lambda: self.lbl_voice_enroll_status.configure(
                    text="✔ 3 / 3 Voice samples enrolled", text_color="#00e676"
                ))
                self.after(0, lambda: messagebox.showinfo("Voice Enrolled", f"Voice model saved for {name}!"))
            self.after(0, lambda: self.btn_record_voice_enroll.configure(state="normal", text="🎤 Record Voice Sample"))
            self.after(0, self.refresh_users_list)

        threading.Thread(target=run_voice, daemon=True).start()

    def refresh_users_list(self):
        self.users_listbox.delete(0, tk.END)
        users = get_all_users()
        for u in users:
            self.users_listbox.insert(tk.END, f"👤 {u['name']} (Enrolled: {u['created_at'][:10]})")

    # ==============================================================
    # PAGE 3: ATTENDANCE LOGS
    # ==============================================================
    def build_logs_page(self):
        page = ctk.CTkFrame(self.pages_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # Header with actions
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=15, sticky="ew")

        ctk.CTkLabel(header, text="Attendance Logs & Reports", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        btn_export = ctk.CTkButton(
            header, text="📥 Export to CSV", command=self.export_csv_action, height=35, fg_color="#1b5e20", hover_color="#004d40"
        )
        btn_export.pack(side="right", padx=5)

        btn_refresh = ctk.CTkButton(
            header, text="🔄 Refresh", command=self.refresh_logs_table, height=35
        )
        btn_refresh.pack(side="right", padx=5)

        # Table Frame
        table_frame = ctk.CTkFrame(page)
        table_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        columns = ("Name", "Date", "Time", "Method", "Face Conf", "Voice Sim", "Status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")

        # Style treeview for dark mode
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", rowheight=28)
        style.configure("Treeview.Heading", background="#1f538d", foreground="white", font=('Arial', 10, 'bold'))

        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        return page

    def refresh_logs_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        logs = get_attendance_logs()
        for r in logs:
            face_conf = f"{r['face_confidence']*100:.1f}%" if r.get('face_confidence') else "N/A"
            voice_sim = f"{r['voice_similarity']*100:.1f}%" if r.get('voice_similarity') else "N/A"
            self.tree.insert("", "end", values=(
                r["user_name"], r["date"], r["time"], r.get("verification_method", "Multi-Modal"),
                face_conf, voice_sim, r.get("status", "Present")
            ))

    def export_csv_action(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"attendance_export_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
        )
        if filepath:
            count = export_attendance_to_csv(filepath)
            messagebox.showinfo("Export Successful", f"Exported {count} records to:\n{filepath}")

    # ==============================================================
    # PAGE 4: SETTINGS & DIAGNOSTICS
    # ==============================================================
    def build_settings_page(self):
        page = ctk.CTkFrame(self.pages_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(page, text="System Settings & Diagnostics", font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(anchor="w", padx=20, pady=(15, 20))

        content = ctk.CTkFrame(page)
        content.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Thresholds
        sec_label = ctk.CTkLabel(content, text="Biometric Thresholds", font=ctk.CTkFont(size=14, weight="bold"))
        sec_label.pack(anchor="w", padx=20, pady=(15, 10))

        # EAR Slider
        ctk.CTkLabel(content, text="Liveness Blink EAR Threshold (0.15 - 0.30):").pack(anchor="w", padx=20)
        self.slider_ear = ctk.CTkSlider(content, from_=0.15, to=0.30, number_of_steps=15)
        self.slider_ear.set(self.liveness_detector.ear_threshold)
        self.slider_ear.pack(fill="x", padx=20, pady=(0, 15))

        # Face Threshold
        ctk.CTkLabel(content, text="Face Similarity Threshold (0.40 - 0.85):").pack(anchor="w", padx=20)
        self.slider_face = ctk.CTkSlider(content, from_=0.40, to=0.85, number_of_steps=20)
        self.slider_face.set(self.face_engine.threshold)
        self.slider_face.pack(fill="x", padx=20, pady=(0, 15))

        # Voice Threshold
        ctk.CTkLabel(content, text="Voice Similarity Threshold (0.35 - 0.80):").pack(anchor="w", padx=20)
        self.slider_voice = ctk.CTkSlider(content, from_=0.35, to=0.80, number_of_steps=20)
        self.slider_voice.set(self.voice_engine.threshold)
        self.slider_voice.pack(fill="x", padx=20, pady=(0, 15))

        btn_save_thresh = ctk.CTkButton(content, text="💾 Apply Thresholds", command=self.apply_thresholds, height=35)
        btn_save_thresh.pack(anchor="w", padx=20, pady=10)

        # Migration section
        ctk.CTkLabel(content, text="Data Migration", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=20, pady=(25, 5))
        btn_migrate = ctk.CTkButton(
            content, text="📦 Sync Legacy Data from images/ and voice_db/",
            command=self.run_migration, height=35, fg_color="#37474f"
        )
        btn_migrate.pack(anchor="w", padx=20, pady=5)

        return page

    def apply_thresholds(self):
        self.liveness_detector.ear_threshold = float(self.slider_ear.get())
        self.face_engine.threshold = float(self.slider_face.get())
        self.voice_engine.threshold = float(self.slider_voice.get())
        messagebox.showinfo("Settings", "Thresholds updated successfully!")

    def run_migration(self):
        def worker():
            count = migrate_legacy_data()
            self.reload_enrolled_data()
            self.after(0, lambda: messagebox.showinfo("Migration Complete", f"Successfully synced {count} profiles into database."))
            self.after(0, self.refresh_users_list)
        threading.Thread(target=worker, daemon=True).start()

    def on_closing(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = BiometricAttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
