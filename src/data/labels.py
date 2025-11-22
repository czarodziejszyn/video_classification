from pathlib import Path
import re


def get_label_from_filename(path):
    """
    Extracts class number from NTU RGB+D skeleton file name.
    """
    filename = Path(path).name

    match = re.search(r"A(\d{3})", filename)

    action_id = int(match.group(1))
    return action_id-1
