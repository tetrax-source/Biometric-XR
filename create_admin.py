"""
create_admin.py
CLI tool to securely create or update an administrator account in system_users.
Prompts for username and password interactively without hardcoding credentials.
"""

import sys
import getpass
from werkzeug.security import generate_password_hash
from core.database import init_db, get_system_user, create_system_user, log_audit, get_connection

def main():
    init_db()
    print("=" * 55)
    print("  BioSecure Attendance - Create Administrator Account")
    print("=" * 55)

    username = input("Enter admin username: ").strip().lower()
    if not username:
        print("[!] Error: Username cannot be empty.")
        sys.exit(1)

    existing = get_system_user(username)
    if existing:
        print(f"[!] User '{username}' already exists with role '{existing['role']}'.")
        choice = input("Do you want to update their password? (y/N): ").strip().lower()
        if choice != 'y':
            print("Cancelled.")
            sys.exit(0)

    password = input("Enter password: ").strip()
    if len(password) < 6:
        print("[!] Error: Password must be at least 6 characters long.")
        sys.exit(1)

    password_confirm = input("Confirm password: ").strip()
    if password != password_confirm:
        print("[!] Error: Passwords do not match.")
        sys.exit(1)

    password_hash = generate_password_hash(password)

    if existing:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE system_users SET password_hash = ? WHERE username = ?", (password_hash, username))
        conn.commit()
        conn.close()
        log_audit(username, "UPDATE_PASSWORD", f"Password updated for {username}")
        print(f"[+] Password updated successfully for admin '{username}'.")
    else:
        create_system_user(username, password_hash, role="admin")
        log_audit(username, "CREATE_ADMIN", f"Admin user created: {username}")
        print(f"[+] Administrator '{username}' created successfully!")

if __name__ == "__main__":
    main()
