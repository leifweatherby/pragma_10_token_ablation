#!/usr/bin/env python3
"""Ensure data is available for the build process.

Checks if data directory exists with content, or if rclone 'pragma' remote
is configured to pull data. Provides helpful error messages for different
scenarios (local development vs CI/CD).
"""

import subprocess
import sys
from pathlib import Path


def setup_service_account_auth() -> bool:
    """Setup rclone to use GCP service account if credentials are available."""
    import os
    import json

    gcp_sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not gcp_sa_json:
        print("✗ GCP_SERVICE_ACCOUNT_JSON not found in environment")
        return False

    print("✓ GCP_SERVICE_ACCOUNT_JSON environment variable detected")
    print(f"  Length: {len(gcp_sa_json)} characters")

    # Validate JSON format
    try:
        sa_data = json.loads(gcp_sa_json)
        if "type" not in sa_data or sa_data["type"] != "service_account":
            print("✗ GCP_SERVICE_ACCOUNT_JSON is not a valid service account key")
            return False
    except json.JSONDecodeError as e:
        print(f"✗ GCP_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")
        return False

    # Configure rclone via environment variables to use service account
    os.environ["RCLONE_CONFIG_PRAGMA_TYPE"] = "drive"
    os.environ["RCLONE_CONFIG_PRAGMA_SCOPE"] = "drive.readonly"
    os.environ["RCLONE_CONFIG_PRAGMA_ROOT_FOLDER_ID"] = "13wRlxC5j7lzkaAA-kA1zFSFeKSGMr73s"
    os.environ["RCLONE_CONFIG_PRAGMA_SERVICE_ACCOUNT_CREDENTIALS"] = gcp_sa_json

    print("✓ Configured rclone to use GCP service account authentication")
    return True


def check_rclone_remote() -> bool:
    """Check if rclone 'pragma' remote is configured."""
    import os

    # Priority 1: Check for GCP service account (CI/CD)
    if setup_service_account_auth():
        return True

    # Priority 2: Check for config file configuration (local dev)
    try:
        result = subprocess.run(
            ["rclone", "listremotes"],
            capture_output=True,
            text=True,
            check=True,
        )
        if "pragma:" in result.stdout:
            print("✓ Found 'pragma:' in rclone config file")
            return True
        else:
            print(f"✗ No 'pragma:' remote found. Available remotes: {result.stdout.strip()}")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Could not run rclone listremotes: {e}")
        return False


def check_data_exists() -> bool:
    """Check if data directory exists and contains files."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    if not data_dir.exists():
        return False

    # Check if directory has any files (not just empty)
    try:
        next(data_dir.iterdir())
        return True
    except StopIteration:
        return False


def pull_data() -> bool:
    """Pull data using rclone."""
    try:
        result = subprocess.run(
            ["pixi", "run", "data-pull"],
            check=True,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    """Ensure data is available, pulling from remote if needed."""
    # First check if data already exists
    if check_data_exists():
        print("✓ Data directory found and contains files")
        return 0

    print("Data directory not found or empty")

    # Check if rclone is configured
    if check_rclone_remote():
        print("✓ rclone 'pragma' remote is configured")
        print("Pulling data from Google Drive...")

        if pull_data():
            print("✓ Data pulled successfully")
            return 0
        else:
            print("\n❌ Error: Failed to pull data from remote")
            print("\nTroubleshooting:")
            print("  - Check your internet connection")
            print("  - Verify rclone authentication hasn't expired")
            print("  - Run 'pixi run data-auth' to re-authenticate")
            return 1

    # No data and no rclone config
    print("\n❌ Error: rclone remote 'pragma' is not configured.")
    print("\nFor local development:")
    print("  Run: pixi run data-auth")
    print("  This will open a browser to authenticate with Google Drive.")
    print("\nFor CI/CD environments:")
    print("  Set GCP service account credentials:")
    print("    GCP_SERVICE_ACCOUNT_JSON=<service-account-json-content>")
    print("\nAlternatively, if you already have data locally:")
    print("  Make sure the 'data/' directory exists and contains run data.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
