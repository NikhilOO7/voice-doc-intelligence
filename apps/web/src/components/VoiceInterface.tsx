// apps/web/src/components/VoiceInterface.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Mic, 
  MicOff, 
  Volume2, 
  Loader, 
  AlertCircle,
  CheckCircle,
  Radio
} from 'lucide-react';
import { 
  Room, 
  RoomEvent, 
  Track, 
  RemoteTrack, 
  LocalAudioTrack,
  ConnectionState
} from 'livekit-client';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    context_level?: string;
    sources?: Array<{
      document_title: string;
      section: string;
      score: number;
    }>;
    latency_ms?: number;
  };
}

interface VoiceMetrics {
  sttLatency: number;
  ttsLatency: number;
  ragLatency: number;
  totalLatency: number;
}

export default function VoiceInterface() {
  // State management
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [contextLevel, setContextLevel] = useState<'local' | 'document' | 'global'>('local');
  const [metrics, setMetrics] = useState<VoiceMetrics | null>(null);
  const [transcript, setTranscript] = useState('');
  
  // LiveKit refs
  const roomRef = useRef<Room | null>(null);
  const audioTrackRef = useRef<LocalAudioTrack | null>(null);
  const conversationIdRef = useRef<string>(`session-${Date.now()}`);
  
  // UI refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  
  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Initialize LiveKit connection
  useEffect(() => {
    initializeLiveKit();
    
    return () => {
      disconnectLiveKit();
    };
  }, []);
  
  const initializeLiveKit = async () => {
    try {
      // Get token from backend
      const response = await fetch('/api/v1/voice/token');
      const { token, url } = await response.json();
      
      // Create room
      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        videoCaptureDefaults: {
          resolution: { width: 0, height: 0 }, // Audio only
        },
      });
      
      // Set up event handlers
      room.on(RoomEvent.Connected, () => {
        console.log('Connected to LiveKit room');
        setIsConnected(true);
        setError(null);
      });
      
      room.on(RoomEvent.Disconnected, () => {
        console.log('Disconnected from LiveKit room');
        setIsConnected(false);
      });
      
      room.on(RoomEvent.DataReceived, (payload: Uint8Array, participant: any) => {
        // Handle data messages from the agent
        const message = JSON.parse(new TextDecoder().decode(payload));
        handleAgentMessage(message);
      });
      
      room.on(RoomEvent.TrackSubscribed, (
        track: RemoteTrack,
        publication: any,
        participant: any
      ) => {
        // Handle incoming audio from agent
        if (track.kind === Track.Kind.Audio) {
          const audioElement = new Audio();
          track.attach(audioElement);
          audioElement.play();
          setIsSpeaking(true);
        }
      });
      
      room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        if (track.kind === Track.Kind.Audio) {
          track.detach();
          setIsSpeaking(false);
        }
      });
      
      room.on(RoomEvent.ConnectionStateChanged, (state: ConnectionState) => {
        console.log('Connection state:', state);
        if (state === ConnectionState.Reconnecting) {
          setError('Reconnecting...');
        }
      });
      
      // Connect to room
      await room.connect(url, token);
      roomRef.current = room;
      
      // Initialize audio context for visualization
      audioContextRef.current = new AudioContext();
      
    } catch (err) {
      console.error('Failed to connect to LiveKit:', err);
      setError('Failed to connect to voice service');
      setIsConnected(false);
    }
  };
  
  const disconnectLiveKit = async () => {
    if (roomRef.current) {
      await roomRef.current.disconnect();
      roomRef.current = null;
    }
    if (audioTrackRef.current) {
      audioTrackRef.current.stop();
      audioTrackRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  };
  
  const handleAgentMessage = (message: any) => {
    switch (message.type) {
      case 'transcript':
        setTranscript(message.text);
        break;
      case 'response':
        addMessage('assistant', message.content, message.metadata);
        setMetrics(message.metrics);
        break;
      case 'error':
        setError(message.error);
        break;
      case 'processing':
        setIsProcessing(message.isProcessing);
        break;
    }
  };
  
  const startRecording = async () => {
    try {
      if (!roomRef.current || !isConnected) {
        setError('Not connected to voice service');
        return;
      }
      
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      
      // Create and publish audio track
      audioTrackRef.current = new LocalAudioTrack(stream.getAudioTracks()[0]);
      await roomRef.current.localParticipant.publishTrack(audioTrackRef.current);
      
      setIsRecording(true);
      setError(null);
      setTranscript('');
      
      // Start visualization
      if (visualizerRef.current && audioContextRef.current) {
        visualizeAudio(stream);
      }
      
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError('Failed to access microphone');
    }
  };
  
  const stopRecording = async () => {
    if (audioTrackRef.current && roomRef.current) {
      roomRef.current.localParticipant.unpublishTrack(audioTrackRef.current);
      audioTrackRef.current.stop();
      audioTrackRef.current = null;
    }
    setIsRecording(false);
    
    // Send the transcript for processing
    if (transcript && roomRef.current) {
      setIsProcessing(true);
      addMessage('user', transcript);
      
      // Send query with context level
      const encoder = new TextEncoder();
      const data = encoder.encode(JSON.stringify({
        type: 'query',
        text: transcript,
        context_level: contextLevel,
        conversation_id: conversationIdRef.current
      }));
      
      await roomRef.current.localParticipant.publishData(
        data,
        { reliable: true }
      );
    }
  };
  
  const visualizeAudio = (stream: MediaStream) => {
    if (!audioContextRef.current || !visualizerRef.current) return;
    
    const source = audioContextRef.current.createMediaStreamSource(stream);
    const analyser = audioContextRef.current.createAnalyser();
    analyser.fftSize = 256;
    
    source.connect(analyser);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const canvas = visualizerRef.current;
    const canvasCtx = canvas.getContext('2d')!;
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;
    
    function draw() {
      if (!isRecording) return;
      
      requestAnimationFrame(draw);
      
      analyser.getByteFrequencyData(dataArray);
      
      canvasCtx.fillStyle = 'rgb(249, 250, 251)';
      canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
      
      const barWidth = (WIDTH / bufferLength) * 2.5;
      let barHeight;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * HEIGHT;
        
        const r = 59;
        const g = 130;
        const b = 246;
        
        canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
        canvasCtx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
    }
    
    draw();
  };
  
  const addMessage = (role: 'user' | 'assistant', content: string, metadata?: any) => {
    const newMessage: Message = {
      id: `msg-${Date.now()}`,
      role,
      content,
      timestamp: new Date(),
      metadata
    };
    setMessages(prev => [...prev, newMessage]);
  };
  
  const sendQuickAction = async (query: string) => {
    if (!roomRef.current || !isConnected) {
      setError('Not connected to voice service');
      return;
    }
    
    addMessage('user', query);
    setIsProcessing(true);
    
    const encoder = new TextEncoder();
    const data = encoder.encode(JSON.stringify({
      type: 'query',
      text: query,
      context_level: contextLevel,
      conversation_id: conversationIdRef.current
    }));
    
    await roomRef.current.localParticipant.publishData(
      data,
      { reliable: true }
    );
  };
  
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-gray-600">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        
        {metrics && (
          <div className="text-xs text-gray-500">
            Latency: {metrics.totalLatency.toFixed(0)}ms
          </div>
        )}
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-3">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <p className="text-sm">{message.content}</p>
              
              {message.metadata?.sources && (
                <div className="mt-2 pt-2 border-t border-gray-200">
                  <p className="text-xs font-medium mb-1">Sources:</p>
                  {message.metadata.sources.map((source, idx) => (
                    <div key={idx} className="text-xs opacity-75">
                      • {source.document_title} - {source.section}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isProcessing && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-2 flex items-center space-x-2">
              <Loader className="w-4 h-4 animate-spin" />
              <span className="text-sm text-gray-600">Thinking...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* Transcript Display */}
      {transcript && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">{transcript}</p>
        </div>
      )}
      
      {/* Audio Visualizer */}
      <canvas
        ref={visualizerRef}
        width={300}
        height={60}
        className={`w-full h-16 mb-4 ${isRecording ? 'block' : 'hidden'}`}
      />
      
      {/* Context Level Selector */}
      <div className="flex items-center justify-center space-x-4 mb-4">
        {(['local', 'document', 'global'] as const).map((level) => (
          <label key={level} className="flex items-center space-x-2 cursor-pointer">
            <input
              type="radio"
              name="context"
              value={level}
              checked={contextLevel === level}
              onChange={(e) => setContextLevel(e.target.value as any)}
              className="text-blue-600"
            />
            <span className="text-sm capitalize text-gray-700">{level}</span>
          </label>
        ))}
      </div>
      
      {/* Voice Controls */}
      <div className="flex flex-col items-center space-y-4">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected || isProcessing}
          className={`relative w-20 h-20 rounded-full flex items-center justify-center transition-all transform hover:scale-105 ${
            isRecording
              ? 'bg-red-500 hover:bg-red-600'
              : 'bg-gradient-to-br from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700'
          } ${(!isConnected || isProcessing) ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isRecording ? (
            <MicOff className="w-8 h-8 text-white" />
          ) : (
            <Mic className="w-8 h-8 text-white" />
          )}
          
          {isRecording && (
            <span className="absolute inset-0 rounded-full border-4 border-red-400 animate-ping" />
          )}
        </button>
        
        {error && (
          <div className="flex items-center space-x-2 text-red-600 text-sm">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}
        
        {isSpeaking && (
          <div className="flex items-center space-x-2 text-blue-600 text-sm">
            <Volume2 className="w-4 h-4" />
            <span>Assistant is speaking...</span>
          </div>
        )}
      </div>
      
      {/* Quick Actions */}
      <div className="mt-6 space-y-2">
        <p className="text-xs text-gray-500 text-center">Quick actions:</p>
        <div className="flex flex-wrap gap-2 justify-center">
          <button
            onClick={() => sendQuickAction("What documents do I have?")}
            className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full"
          >
            List documents
          </button>
          <button
            onClick={() => sendQuickAction("Summarize recent uploads")}
            className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full"
          >
            Recent summary
          </button>
          <button
            onClick={() => sendQuickAction("Find action items")}
            className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full"
          >
            Action items
          </button>
        </div>
      </div>
    </div>
  );
}