from pathlib import Path
from cryptography.fernet import Fernet
import MetaTrader5 as mt5

# Global variable to track login status. Ensure it's defined before calling login_mt5.
mt5_logged_in = False

#-----------------------------------------------------------------------------------------
#  MT5 Utilities
#-----------------------------------------------------------------------------------------

def encrypt_credentials(login: str, password: str, server: str, filepath: str) -> None:
    """
    Encrypt MT5 credentials and write them to a file.

    This function uses Fernet symmetric encryption from the cryptography library to secure the 
    provided credentials. A random encryption key is generated, and that key along with the 
    encrypted login, password, and server are saved to the specified file. The key and encrypted 
    values are stored on separate lines (binary format).
    NOTE: Ensure that the file path is not stored in a publicly accessible location.

    Args:
        login (str): The MT5 login credential. Note: even if the login is numeric, pass it as a string.
        password (str): The account password.
        server (str): The server address.
        filepath (str): The full path to the file where encrypted credentials will be stored.

    Returns:
        None

    Raises:
        IOError: If there is any issue writing data to the specified file.
    """
    # Generate a random encryption key and create a cipher suite
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    # Encrypt each credential individually
    encrypted_login = cipher_suite.encrypt(login.encode())
    encrypted_password = cipher_suite.encrypt(password.encode())
    encrypted_server = cipher_suite.encrypt(server.encode())

    # Combine all parts with a newline separator (in binary) so each element is on its own line.
    data = b'\n'.join([key, encrypted_login, encrypted_password, encrypted_server])

    try:
        with open(filepath, 'wb') as file:
            file.write(data)
    except IOError as e:
        print(f"Error writing credentials to file '{filepath}': {e}")
        raise  # Re-raise the exception for upstream handling


def decrypt_credentials(filepath: str) -> tuple:
    """
    Decrypt and retrieve MT5 credentials from a file.

    Reads encrypted credentials from the specified file. The file is expected to have the encryption
    key on the first line, followed by three lines containing the encrypted login, password, and server.
    After decryption, if the login value is numeric it is returned as an int; otherwise, it remains a string.

    Args:
        filepath (str): The full path to the file containing the encrypted credentials.

    Returns:
        tuple: A tuple containing (login, password, server). 'login' is converted to int if numeric.

    Raises:
        IOError: If the file cannot be read.
        Exception: If decryption fails.
    """
    try:
        with open(filepath, 'rb') as file:
            key = file.readline().strip()  # First line: decryption key
            encrypted_login = file.readline().strip()
            encrypted_password = file.readline().strip()
            encrypted_server = file.readline().strip()
    except IOError as e:
        print(f"Error reading credentials from file '{filepath}': {e}")
        raise

    cipher_suite = Fernet(key)

    # Decrypt the credentials and decode them from bytes to string
    try:
        login = cipher_suite.decrypt(encrypted_login).decode()
        password = cipher_suite.decrypt(encrypted_password).decode()
        server = cipher_suite.decrypt(encrypted_server).decode()
    except Exception as e:
        print(f"Error decrypting credentials: {e}")
        raise

    # Convert login to int if it represents a numeric value (as required by MT5)
    if login.isnumeric():
        login = int(login)

    return login, password, server


def login_mt5(account: str, timeout: int = 30000, verbose: bool = True) -> None:
    """
    Log in to a MetaTrader5 account using encrypted credentials.

    This function retrieves encrypted credentials from a predefined file and attempts 
    to initialize a connection to MetaTrader5 using the decrypted details. On a successful connection, 
    and if verbose is True, it prints detailed information about the connection, model package, and 
    terminal status. In case of failure the function shuts down the connection and reports an error.

    Args:
        account (str): Account name used to identify the credentials file (expected at 'credentials\{account}.txt').
        timeout (int): Connection timeout in milliseconds.
        verbose (bool): Whether to display detailed connection and diagnostics information.

    Returns:
        None

    Side Effects:
        - Updates the global variable `mt5_logged_in` to reflect connection status.
        - Prints status and diagnostic information.
    """
    global mt5_logged_in

    credentials_path = fr"credentials\{account}.txt"

    # Attempt to load and decrypt credentials
    try:
        login, password, server = decrypt_credentials(credentials_path)
    except Exception as e:
        print(f"Failed to retrieve credentials for account '{account}': {e}")
        mt5_logged_in = False
        return

    # Path to the MT5 terminal executable (update this path according to your installation)
    MT5_path = r'C:\Program Files\MetaTrader 5\terminal64.exe'

    # Initialize connection to MT5 using the decrypted credentials
    if not mt5.initialize(login=login, password=password, server=server,
                          timeout=timeout, path=MT5_path):
        print("MT5 initialize() failed, error code =", mt5.last_error())
        mt5.shutdown()
        mt5_logged_in = False
        print(f"Failed to login to MT5 using {account} account")
        return

    mt5_logged_in = True
    print(f"Logged in to MT5 as {account}")

    if verbose:
        # Print MetaTrader5 package details and connection diagnostics
        print("MetaTrader5 package author:", mt5.__author__)
        print("MetaTrader5 package version:", mt5.__version__)
        print(f"MT5 version: {mt5.version()}")

        terminal_info = mt5.terminal_info()._asdict()
        print("MT5 terminal information:")
        for key, value in terminal_info.items():
            print(f"\t{key} = {value}")

        # Compute and print the data path for files saved from the terminal
        MT5_DATA_PATH = Path(terminal_info.get('data_path', ''), 'MQL5', 'Files')
        print(f"Files saved from MT5 Terminal are located at: {MT5_DATA_PATH}")
