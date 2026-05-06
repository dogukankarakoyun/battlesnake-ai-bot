import datetime

DEBUG = True


def debug(msg):
    """
    Print a timestamped debug message if debugging is enabled.
    """
    if DEBUG:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)