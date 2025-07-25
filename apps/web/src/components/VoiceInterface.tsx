// apps/web/src/components/VoiceInterface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { 
  Room, 
  RoomEvent, 
  RemoteParticipant, 
  RemoteTrackPublication,
  RemoteTrack,
  TrackKind
} from 'livekit-client';

interface VoiceInterfaceProps {
  className?: string;
}

interface RoomInfo {
  room_name: string;
  token: string;
  url: string;
  participant_name: string;
}

export default function VoiceInterface({ className = "" }: VoiceInterfaceProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const roomRef = useRef<Room | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  
  // Create audio element for playback
  useEffect(() => {
    if (!audioElementRef.current) {
      audioElementRef.current = document.createElement('audio');
      audioElementRef.current.autoplay = true;
      audioElementRef.current.controls = false;
    }
  }, []);
  
  const createVoiceRoom = async (): Promise<RoomInfo> => {
    const response = await fetch('/api/v1/voice/create-room', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        participant_name: 'User'
      })
    });
    
    if (!response.ok) {
      throw new Error('Failed to create voice room');
    }
    
    return await response.json();
  };
  
  const connectToVoice = async () => {
    if (isConnected || isConnecting) return;
    
    setIsConnecting(true);
    setError(null);
    
    try {
      // Create room and get connection info
      const roomInfo = await createVoiceRoom();
      
      // Create and connect to LiveKit room
      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        publishDefaults: {
          audioPreset: {
            maxBitrate: 64000,
          }
        }
      });
      
      roomRef.current = room;
      
      // Set up event listeners
      room.on(RoomEvent.Connected, () => {
        console.log('Connected to voice room');
        setIsConnected(true);
        setIsConnecting(false);
        addTranscript('🎙️ Connected to voice assistant. You can start speaking!');
      });
      
      room.on(RoomEvent.Disconnected, () => {
        console.log('Disconnected from voice room');
        setIsConnected(false);
        setIsConnecting(false);
        addTranscript('❌ Disconnected from voice assistant');
      });
      
      room.on(RoomEvent.ParticipantConnected, (participant: RemoteParticipant) => {
        console.log('Agent joined:', participant.identity);
        addTranscript('🤖 Voice assistant joined the conversation');
      });
      
      room.on(RoomEvent.TrackSubscribed, (
        track: RemoteTrack,
        publication: RemoteTrackPublication,
        participant: RemoteParticipant
      ) => {
        if (track.kind === TrackKind.Audio && audioElementRef.current) {
          track.attach(audioElementRef.current);
          console.log('Subscribed to agent audio track');
        }
      });
      
      // Handle transcriptions (if supported)
      room.on(RoomEvent.DataReceived, (payload: Uint8Array, participant?: RemoteParticipant) => {
        try {
          const data = JSON.parse(new TextDecoder().decode(payload));
          if (data.type === 'transcription') {
            const speaker = participant?.identity === 'agent' ? '🤖 Assistant' : '👤 You';
            addTranscript(`${speaker}: ${data.text}`);
          }
        } catch (e) {
          // Ignore invalid JSON
        }
      });
      
      room.on(RoomEvent.ConnectionStateChanged, (state) => {
        console.log('Connection state changed:', state);
        if (state === 'failed' || state === 'disconnected') {
          setError('Connection lost. Please try reconnecting.');
          setIsConnected(false);
          setIsConnecting(false);
        }
      });
      
      // Connect to the room
      await room.connect(roomInfo.url, roomInfo.token);
      
      // Enable microphone
      await room.localParticipant.enableMicrophone(true);
      
    } catch (err) {
      console.error('Failed to connect:', err);
      setError(err instanceof Error ? err.message : 'Failed to connect to voice assistant');
      setIsConnecting(false);
    }
  };
  
  const disconnectFromVoice = async () => {
    if (roomRef.current) {
      await roomRef.current.disconnect();
      roomRef.current = null;
    }
    setIsConnected(false);
    setIsConnecting(false);
  };
  
  const addTranscript = (message: string) => {
    setTranscript(prev => [...prev.slice(-9), message]); // Keep last 10 messages
  };
  
  // Toggle microphone
  const toggleMicrophone = async () => {
    if (roomRef.current) {
      const enabled = roomRef.current.localParticipant.isMicrophoneEnabled;
      await roomRef.current.localParticipant.enableMicrophone(!enabled);
    }
  };
  
  // Check if microphone is enabled
  const isMicrophoneEnabled = roomRef.current?.localParticipant.isMicrophoneEnabled ?? false;
  
  return (
    <div className={`voice-interface ${className}`}>
      <div className="voice-controls">
        <div className="connection-status">
          {isConnecting ? (
            <div className="status connecting">
              <div className="spinner"></div>
              <span>Connecting...</span>
            </div>
          ) : isConnected ? (
            <div className="status connected">
              <div className="pulse-dot"></div>
              <span>Voice Assistant Ready</span>
            </div>
          ) : (
            <div className="status disconnected">
              <div className="offline-dot"></div>
              <span>Not Connected</span>
            </div>
          )}
        </div>
        
        <div className="voice-buttons">
          {!isConnected ? (
            <button 
              onClick={connectToVoice}
              disabled={isConnecting}
              className="connect-btn"
            >
              {isConnecting ? 'Connecting...' : '🎙️ Start Voice Chat'}
            </button>
          ) : (
            <div className="active-controls">
              <button 
                onClick={toggleMicrophone}
                className={`mic-btn ${isMicrophoneEnabled ? 'enabled' : 'disabled'}`}
              >
                {isMicrophoneEnabled ? '🎤' : '🎤❌'}
              </button>
              
              <button 
                onClick={disconnectFromVoice}
                className="disconnect-btn"
              >
                ❌ End Chat
              </button>
            </div>
          )}
        </div>
      </div>
      
      {error && (
        <div className="error-message">
          ⚠️ {error}
          <button onClick={() => setError(null)} className="close-error">×</button>
        </div>
      )}
      
      {isConnected && (
        <div className="conversation-transcript">
          <h3>Conversation</h3>
          <div className="transcript-messages">
            {transcript.map((message, index) => (
              <div key={index} className="transcript-message">
                {message}
              </div>
            ))}
          </div>
          
          <div className="voice-tips">
            <p><strong>💡 Try saying:</strong></p>
            <ul>
              <li>"Search for safety protocols"</li>
              <li>"What documents do I have?"</li>
              <li>"Summarize the latest report"</li>
              <li>"Find information about project requirements"</li>
            </ul>
          </div>
        </div>
      )}
      
      <style jsx>{`
        .voice-interface {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 12px;
          padding: 24px;
          color: white;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .voice-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        
        .status {
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 500;
        }
        
        .pulse-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #4ade80;
          animation: pulse 2s infinite;
        }
        
        .offline-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #ef4444;
        }
        
        .spinner {
          width: 12px;
          height: 12px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top: 2px solid white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .voice-buttons button {
          background: rgba(255,255,255,0.2);
          border: 1px solid rgba(255,255,255,0.3);
          color: white;
          padding: 10px 20px;
          border-radius: 8px;
          cursor: pointer;
          font-weight: 500;
          transition: all 0.2s;
        }
        
        .voice-buttons button:hover:not(:disabled) {
          background: rgba(255,255,255,0.3);
          transform: translateY(-1px);
        }
        
        .voice-buttons button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        .active-controls {
          display: flex;
          gap: 12px;
        }
        
        .mic-btn.enabled {
          background: rgba(74, 222, 128, 0.3);
          border-color: rgba(74, 222, 128, 0.5);
        }
        
        .mic-btn.disabled {
          background: rgba(239, 68, 68, 0.3);
          border-color: rgba(239, 68, 68, 0.5);
        }
        
        .error-message {
          background: rgba(239, 68, 68, 0.2);
          border: 1px solid rgba(239, 68, 68, 0.3);
          padding: 12px;
          border-radius: 8px;
          margin-bottom: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .close-error {
          background: none;
          border: none;
          color: white;
          cursor: pointer;
          font-size: 18px;
          padding: 0;
        }
        
        .conversation-transcript {
          background: rgba(255,255,255,0.1);
          border-radius: 8px;
          padding: 16px;
        }
        
        .transcript-messages {
          max-height: 200px;
          overflow-y: auto;
          margin-bottom: 16px;
          border-bottom: 1px solid rgba(255,255,255,0.2);
          padding-bottom: 16px;
        }
        
        .transcript-message {
          margin-bottom: 8px;
          padding: 6px 0;
          opacity: 0.9;
          line-height: 1.4;
        }
        
        .voice-tips {
          opacity: 0.8;
          font-size: 14px;
        }
        
        .voice-tips ul {
          margin: 8px 0 0 0;
          padding-left: 20px;
        }
        
        .voice-tips li {
          margin-bottom: 4px;
        }
      `}</style>
    </div>
  );
}