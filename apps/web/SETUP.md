# Frontend Setup Guide

Quick start guide for setting up and running the Voice Document Intelligence frontend.

## Step 1: Install Node.js

### macOS
```bash
# Using Homebrew
brew install node

# Or download from https://nodejs.org/
```

### Linux (Ubuntu/Debian)
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Verify Installation
```bash
node --version  # Should show v18.0.0 or higher
npm --version   # Should show 9.0.0 or higher
```

## Step 2: Install Dependencies

```bash
# Navigate to the frontend directory
cd /Users/farheenzubair/Documents/voice-doc-intelligence/apps/web

# Install all dependencies
npm install
```

This will install:
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- LiveKit React Components
- Axios
- And all other dependencies

## Step 3: Configure Environment

The `.env.local` file is already created with default values:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
```

**Update these if your backend or LiveKit runs on different ports!**

## Step 4: Start the Development Server

```bash
npm run dev
```

The frontend will start on [http://localhost:3000](http://localhost:3000)

## Step 5: Verify Everything Works

### Check the UI
1. Open http://localhost:3000 in your browser
2. You should see the Voice Document Intelligence interface
3. Check the health status indicator in the top-right (should be green if backend is running)

### Test Document Upload
1. Go to the "Documents" tab
2. Drag and drop a PDF or DOCX file
3. You should see it processing
4. Once complete, it will show "completed" status

### Test Chat
1. Go to the "Chat" tab
2. Type a question about your uploaded document
3. You should get an AI-generated response with sources

### Test Voice (Optional)
1. Go to the "Voice" tab
2. Click "Connect to Voice Assistant"
3. Allow microphone permissions
4. Speak your question

### Check Analytics
1. Go to the "Analytics" tab
2. You should see metrics and statistics

## Common Issues

### Port 3000 Already in Use

```bash
# Kill the process using port 3000
lsof -ti:3000 | xargs kill -9

# Or run on a different port
PORT=3001 npm run dev
```

### Cannot Connect to Backend

**Error:** "Failed to load documents" or red health status

**Solutions:**
1. Ensure backend is running:
   ```bash
   cd /Users/farheenzubair/Documents/voice-doc-intelligence
   # Start backend (see backend README)
   ```

2. Check backend is accessible:
   ```bash
   curl http://localhost:8000/health
   ```

3. Verify CORS is enabled in backend

### LiveKit Voice Not Working

**Error:** "Failed to connect to voice service"

**Solutions:**
1. Ensure LiveKit server is running:
   ```bash
   docker-compose up livekit
   ```

2. Check LiveKit is accessible:
   ```bash
   curl http://localhost:7880
   ```

3. Verify microphone permissions in browser

### Module Not Found Errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### TypeScript Errors

```bash
# Type check without running
npm run type-check

# Most errors can be fixed by ensuring dependencies are installed
npm install
```

## Production Build

To create a production build:

```bash
# Build
npm run build

# Start production server
npm start
```

## Quick Start Checklist

- [ ] Node.js 18+ installed
- [ ] Dependencies installed (`npm install`)
- [ ] Backend running on port 8000
- [ ] LiveKit running on port 7880 (for voice features)
- [ ] Docker services running (Postgres, Redis, Qdrant)
- [ ] Environment variables configured
- [ ] Development server started (`npm run dev`)
- [ ] Browser opened to http://localhost:3000

## Next Steps

1. **Upload Documents**: Start by uploading some PDF or DOCX files
2. **Try Chat**: Ask questions about your documents
3. **Explore Voice**: Connect to the voice assistant
4. **Monitor Analytics**: Check the analytics dashboard

## Getting Help

If you encounter issues:
1. Check the browser console (F12) for errors
2. Check the terminal where `npm run dev` is running
3. Verify all backend services are running
4. Check the backend logs for API errors

## Development Tips

- Use browser DevTools (F12) to debug
- Check Network tab for failed API calls
- Use React DevTools extension for component debugging
- Hot reload is enabled - changes reflect immediately
- TypeScript errors show in the terminal and browser

## File Structure Reference

```
apps/web/
├── app/                 # Next.js pages
├── components/          # React components
├── lib/                # Utilities & API client
├── types/              # TypeScript types
├── public/             # Static assets
├── package.json        # Dependencies
├── tsconfig.json       # TypeScript config
├── tailwind.config.ts  # Tailwind config
└── next.config.js      # Next.js config
```

Ready to start! Run `npm run dev` and open http://localhost:3000
