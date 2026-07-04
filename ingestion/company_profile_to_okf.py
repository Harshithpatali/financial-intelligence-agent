from pathlib import Path
import shutil

def copy_company_profile():

    source = Path(
        "data/company_info/tcs_profile.txt"
    )

    destination_dir = Path(
        "knowledge/company_profile"
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        destination_dir /
        "tcs_profile.md"
    )

    shutil.copy(
        source,
        destination
    )

    print(
        "Company profile copied"
    )