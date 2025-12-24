# Railway Deployment Guide

## Quick Deploy (Easiest Method)

### Option 1: Deploy via GitHub (Recommended)

1. **Create a GitHub repository**
   - Go to https://github.com/new
   - Name it: `wordle-scorer`
   - Make it private or public (your choice)
   - Don't initialize with README (we have files already)

2. **Upload your files to GitHub**
   ```bash
   cd /path/to/your/wordle/files
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/wordle-scorer.git
   git push -u origin main
   ```

3. **Deploy on Railway**
   - Go to https://railway.app
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your GitHub
   - Select your `wordle-scorer` repository
   - Railway will automatically detect Python and deploy!
   - Wait 2-3 minutes for build to complete

4. **Get your URL**
   - Click on your project
   - Go to "Settings" tab
   - Click "Generate Domain"
   - You'll get a URL like: `wordle-scorer-production.up.railway.app`

---

### Option 2: Deploy via Railway CLI

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   # OR
   brew install railway
   ```

2. **Login and Deploy**
   ```bash
   cd /path/to/your/wordle/files
   railway login
   railway init
   railway up
   ```

3. **Generate Domain**
   ```bash
   railway domain
   ```

---

## Files You Need

Make sure you have these files in your project folder:
- `app.py` - Flask server
- `wordle_scorer.py` - OCR scoring logic
- `wordle_scorer.html` - Frontend UI
- `requirements.txt` - Python dependencies
- `nixpacks.toml` - Build configuration (tells Railway to install Tesseract)
- `Procfile` - Start command

All these files are included in your download!

---

## After Deployment

Once deployed, you can:
- Share the URL with your husband and friends
- Everyone can upload Wordle screenshots
- Automatic scoring with OCR
- No installation needed for users

---

## Troubleshooting

**Build fails:**
- Check logs in Railway dashboard
- Ensure all files are uploaded
- Make sure `nixpacks.toml` is present (installs Tesseract)

**App crashes:**
- Check Runtime logs in Railway
- Verify environment variables if needed

**Can't access the site:**
- Make sure domain is generated in Settings
- Check if service is running (should show green dot)

---

## Cost

Railway free tier includes:
- $5 credit per month
- Enough for personal use
- No credit card required for trial

This app uses minimal resources, so free tier should be plenty!

---

## Alternative: Render.com

If Railway doesn't work, try Render.com:

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect GitHub repo
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - Add package: `tesseract-ocr` (in Native Environment section)
5. Deploy!

---

Need help? The deployment should be straightforward with either method!
