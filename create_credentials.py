"""
Helper script to create encrypted MT5 credentials file.

Usage:
    python create_credentials.py

This script will prompt you for your MT5 login, password, server, and account name,
then create an encrypted credentials file in the credentials folder.
"""

from mt5_login import encrypt_credentials
from pathlib import Path

def main():
    print("=" * 60)
    print("MT5 Credentials Encryption Tool")
    print("=" * 60)
    print("\nThis tool will encrypt your MT5 credentials and save them securely.\n")
    
    # Get user input
    login = input("Enter your MT5 login (account number): ").strip()
    password = input("Enter your MT5 password: ").strip()
    server = input("Enter your MT5 server (e.g., MetaQuotes-Demo): ").strip()
    account_name = input("Enter account name for this file (e.g., demo, live): ").strip()
    
    # Validate inputs
    if not all([login, password, server, account_name]):
        print("\nError: All fields are required!")
        return
    
    # Create credentials directory if it doesn't exist
    credentials_dir = Path("credentials")
    credentials_dir.mkdir(exist_ok=True)
    
    # Generate filepath
    filepath = credentials_dir / f"{account_name}.txt"
    
    # Encrypt and save
    try:
        encrypt_credentials(login, password, server, str(filepath))
        print(f"\n✓ Success! Credentials encrypted and saved to: {filepath}")
        print(f"\nYou can now use: login_mt5('{account_name}') to connect to MT5")
    except Exception as e:
        print(f"\n✗ Error: Failed to encrypt credentials: {e}")

if __name__ == "__main__":
    main()
