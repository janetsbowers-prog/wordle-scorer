#!/usr/bin/env bash
set -o errexit

# Update package lists
apt-get update -y

# Install Tesseract OCR
apt-get install -y tesseract-ocr

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Make sure it has executable permissions** by also creating a file called `.gitattributes` with:
```
render-build.sh text eol=lf
