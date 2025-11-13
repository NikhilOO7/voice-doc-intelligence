// API Client for Voice Document Intelligence System
import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  Document,
  QueryRequest,
  QueryResponse,
  HealthStatus,
  UsageAnalytics,
  PerformanceAnalytics,
  VoiceToken,
} from '@/types';

class ApiClient {
  private client: AxiosInstance;
  private wsUrl: string;

  constructor() {
    const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    this.wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`[API] Response:`, response.status);
        return response;
      },
      (error: AxiosError) => {
        console.error(`[API] Error:`, error.response?.status, error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health Check
  async healthCheck(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  // Document Management
  async uploadDocument(file: File, useEnhanced: boolean = true): Promise<{ document_id: string; filename: string; status: string; message: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post(
      `/api/v1/documents/upload?use_enhanced=${useEnhanced}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  async listDocuments(skip: number = 0, limit: number = 10, status?: string): Promise<{
    documents: Document[];
    total: number;
    skip: number;
    limit: number;
  }> {
    const params: any = { skip, limit };
    if (status) params.status = status;

    const response = await this.client.get('/api/v1/documents', { params });
    return response.data;
  }

  async getDocument(documentId: string): Promise<Document> {
    const response = await this.client.get<Document>(`/api/v1/documents/${documentId}`);
    return response.data;
  }

  async deleteDocument(documentId: string): Promise<{ message: string }> {
    const response = await this.client.delete(`/api/v1/documents/${documentId}`);
    return response.data;
  }

  // Query
  async queryDocuments(request: QueryRequest): Promise<QueryResponse> {
    const response = await this.client.post<QueryResponse>('/api/v1/query', request);
    return response.data;
  }

  // Voice
  async getVoiceToken(roomName: string = 'document-chat', participantName: string = 'user'): Promise<VoiceToken> {
    const response = await this.client.get<VoiceToken>('/api/v1/voice/token', {
      params: { room_name: roomName, participant_name: participantName },
    });
    return response.data;
  }

  // Analytics
  async getUsageAnalytics(): Promise<UsageAnalytics> {
    const response = await this.client.get<UsageAnalytics>('/api/v1/analytics/usage');
    return response.data;
  }

  async getPerformanceAnalytics(): Promise<PerformanceAnalytics> {
    const response = await this.client.get<PerformanceAnalytics>('/api/v1/analytics/performance');
    return response.data;
  }

  // Test
  async testDemo(): Promise<any> {
    const response = await this.client.get('/api/v1/test/demo');
    return response.data;
  }

  async processSampleDocument(): Promise<{ document_id: string; message: string }> {
    const response = await this.client.post('/api/v1/test/process-sample');
    return response.data;
  }

  // WebSocket
  createWebSocket(onMessage: (message: any) => void, onError?: (error: Event) => void): WebSocket {
    const ws = new WebSocket(`${this.wsUrl}/ws`);

    ws.onopen = () => {
      console.log('[WebSocket] Connected');
      // Send ping every 30 seconds to keep connection alive
      setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        // Ignore pong responses from keepalive ping
        if (event.data === 'pong') {
          return;
        }
        const message = JSON.parse(event.data);
        onMessage(message);
      } catch (error) {
        console.error('[WebSocket] Failed to parse message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
      if (onError) onError(error);
    };

    ws.onclose = () => {
      console.log('[WebSocket] Disconnected');
    };

    return ws;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
