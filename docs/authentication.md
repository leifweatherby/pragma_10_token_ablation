# Data Authentication

This document explains how to authenticate with Google Drive to access the pragma dataset.

## Overview

The application requires access to training data stored in Google Drive. Two authentication methods are available:

1. **OAuth (Local Development)** - Interactive browser-based authentication
2. **GCP Service Account (CI/CD)** - Automated authentication using service account credentials

## Authentication Priority

When the application starts, it checks for data access in this order:

1. **Existing data**: If `data/` directory exists and contains files, use it
2. **Service account**: If `GCP_SERVICE_ACCOUNT_JSON` environment variable is set
3. **OAuth token**: If OAuth token exists in rclone config file
4. **Error**: If none available, show setup instructions

## Local Development (OAuth)

### Setup

1. Run authentication:
   ```bash
   pixi run data-auth
   ```

2. Complete browser authentication (sign in, grant Drive permissions)

3. Pull the data:
   ```bash
   pixi run data-pull
   ```

OAuth tokens are stored in `~/.config/rclone/rclone.conf` and auto-refresh.

## CI/CD (Service Account)

### Setup

1. **Create a GCP Service Account** (Project administrators only):
   - Create service account with Google Drive API permissions
   - Download the JSON key file

2. **Share Google Drive Folder** with the service account email

3. **Configure Environment Variable**:
   - Set `GCP_SERVICE_ACCOUNT_JSON` to the JSON key file contents
   - For CloudFlare Pages: Add as encrypted environment variable
   - For other CI/CD: Set as a secret

### Security Best Practices

- Never commit service account keys to the repository
- Store keys only in encrypted environment variables
- Grant minimum necessary permissions (read-only)
- Rotate keys periodically

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Data pull fails | Check internet, permissions, and API quotas |
| Invalid JSON | Ensure `GCP_SERVICE_ACCOUNT_JSON` contains valid JSON |
| Wrong key type | Confirm JSON has `"type": "service_account"` |
| Token expired | Run `pixi run data-auth` again |
| Variable not detected | Check spelling: `GCP_SERVICE_ACCOUNT_JSON` exactly |

## Summary

| Method | Best For | Setup |
|--------|----------|-------|
| OAuth | Local development | `pixi run data-auth` |
| Service Account | CI/CD | GCP setup + env variable |
