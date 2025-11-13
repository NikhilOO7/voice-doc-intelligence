# Voice Document Intelligence - Frontend

Modern React + TypeScript frontend for the Voice Document Intelligence System built with Next.js 14.

## Features

- **Document Management**: Upload, view, and delete documents with real-time processing status
- **Smart Chat Interface**: Query documents using natural language with contextual search
- **Voice Assistant**: Talk to your documents using LiveKit-powered voice interface
- **Analytics Dashboard**: Monitor usage, performance metrics, and system health
- **Real-time Updates**: WebSocket integration for live document processing notifications

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Voice**: LiveKit React Components
- **HTTP Client**: Axios
- **Icons**: Lucide React
- **Charts**: Recharts (for analytics)

## Prerequisites

Before running the frontend, make sure you have:

- Node.js 18+ and npm/yarn/pnpm
- Backend API running on `http://localhost:8000`
- LiveKit server running on `ws://localhost:7880`

## Installation

```bash
# Navigate to the frontend directory
cd apps/web

# Install dependencies (choose one)
npm install
# or
yarn install
# or
pnpm install
```

## Environment Variables

Create a `.env.local` file in the `apps/web` directory:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880

# Feature Flags
NEXT_PUBLIC_ENABLE_VOICE=true
NEXT_PUBLIC_ENABLE_ANALYTICS=true
```

## Development

```bash
# Start the development server
npm run dev
# or
yarn dev
# or
pnpm dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)

## Project Structure

```
apps/web/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main application page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── documents/         # Document management components
│   ├── chat/             # Chat interface components
│   ├── voice/            # Voice interface components
│   └── analytics/        # Analytics dashboard components
├── lib/                  # Utility libraries
│   ├── api-client.ts    # API client
│   └── utils.ts         # Utility functions
├── types/               # TypeScript type definitions
│   └── index.ts        # Main types
└── public/             # Static assets
```

## Key Components

### Document Management
- `DocumentUpload.tsx`: Drag-and-drop document upload with progress tracking
- `DocumentList.tsx`: List of uploaded documents with status and actions

### Chat Interface
- `ChatInterface.tsx`: Full-featured chat UI with context level selection
- Displays source documents and relevance scores
- Shows query latency metrics

### Voice Interface
- `VoiceInterface.tsx`: LiveKit-powered voice assistant
- Real-time audio visualization
- Voice activity detection
- Natural conversation flow

### Analytics Dashboard
- `AnalyticsDashboard.tsx`: Comprehensive metrics dashboard
- Document processing statistics
- Performance latency metrics
- Success rate tracking

## API Integration

The frontend communicates with the backend through:

1. **REST API** (`/api/v1/*`):
   - Document operations (upload, list, delete)
   - Query processing
   - Analytics retrieval
   - Voice token generation

2. **WebSocket** (`/ws`):
   - Real-time document processing updates
   - Status change notifications

3. **LiveKit** (WebRTC):
   - Real-time voice communication
   - Audio streaming

## Features in Detail

### 3-Level Contextual Search
Choose from three context levels when querying:
- **Local**: Search within nearby chunks
- **Document**: Search across the entire document
- **Global**: Search across all documents

### Enhanced Processing
Documents can be processed with:
- **Basic**: Standard chunking and embeddings
- **Enhanced**: 3-level contextual embeddings with intelligent chunking

### Real-time Notifications
- Browser notifications for completed processing
- In-app toast notifications
- Live document list updates

## Building for Production

```bash
# Create production build
npm run build

# Start production server
npm start
```

## Type Safety

The project uses TypeScript for full type safety:
- All API responses are typed
- Component props are strictly typed
- Utility functions have proper type annotations

## Performance Optimizations

- Lazy loading of heavy components
- Optimized re-renders with React hooks
- Debounced search inputs
- Image and asset optimization
- Code splitting by route

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Troubleshooting

### Cannot connect to API
- Ensure backend is running on port 8000
- Check CORS settings in backend
- Verify `NEXT_PUBLIC_API_URL` in `.env.local`

### Voice not working
- Check LiveKit server is running
- Verify microphone permissions
- Check `NEXT_PUBLIC_LIVEKIT_URL` configuration

### WebSocket not connecting
- Ensure backend WebSocket endpoint is accessible
- Check firewall/proxy settings
- Verify `NEXT_PUBLIC_WS_URL` configuration

## Contributing

1. Follow the existing code structure
2. Use TypeScript for all new files
3. Follow the component naming conventions
4. Add proper types for all API interactions
5. Test responsiveness on mobile devices

## License

Part of the Voice Document Intelligence System
