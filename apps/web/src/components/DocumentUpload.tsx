// apps/web/src/components/DocumentUpload.tsx
import React, { useState, useCallback } from 'react';
import { Upload, File, CheckCircle, AlertCircle, Loader } from 'lucide-react';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
}

export default function DocumentUpload() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      handleFiles(selectedFiles);
    }
  }, []);

  const handleFiles = async (fileList: File[]) => {
    const validFiles = fileList.filter(file => {
      const validTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/markdown'
      ];
      return validTypes.includes(file.type) && file.size <= 100 * 1024 * 1024; // 100MB
    });

    const newFiles: UploadedFile[] = validFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      status: 'uploading',
      progress: 0
    }));

    setFiles(prev => [...prev, ...newFiles]);

    // Upload each file
    for (const [index, file] of validFiles.entries()) {
      await uploadFile(file, newFiles[index].id);
    }
  };

  const uploadFile = async (file: File, fileId: string) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('metadata', JSON.stringify({
        category: 'user_upload',
        upload_time: new Date().toISOString()
      }));

      // Simulate upload progress
      const updateProgress = (progress: number) => {
        setFiles(prev => prev.map(f => 
          f.id === fileId ? { ...f, progress } : f
        ));
      };

      // Start upload
      updateProgress(10);

      const response = await fetch('/api/v1/documents/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      updateProgress(100);

      // Update status to processing
      setFiles(prev => prev.map(f => 
        f.id === fileId ? { ...f, status: 'processing', progress: 100 } : f
      ));

      // Poll for processing status
      await pollProcessingStatus(result.document_id, fileId);

    } catch (error) {
      console.error('Upload failed:', error);
      setFiles(prev => prev.map(f => 
        f.id === fileId ? { 
          ...f, 
          status: 'error', 
          error: error instanceof Error ? error.message : 'Upload failed'
        } : f
      ));
    }
  };

  const pollProcessingStatus = async (documentId: string, fileId: string) => {
    const maxAttempts = 30; // 30 attempts with 2s interval = 1 minute
    let attempts = 0;

    const poll = async () => {
      try {
        const response = await fetch(`/api/v1/documents/${documentId}`);
        if (!response.ok) throw new Error('Status check failed');

        const data = await response.json();
        
        if (data.processing_status === 'completed') {
          setFiles(prev => prev.map(f => 
            f.id === fileId ? { ...f, status: 'completed' } : f
          ));
          return;
        }
        
        if (data.processing_status === 'failed') {
          setFiles(prev => prev.map(f => 
            f.id === fileId ? { 
              ...f, 
              status: 'error', 
              error: data.error?.error || 'Processing failed'
            } : f
          ));
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 2000); // Poll every 2 seconds
        } else {
          setFiles(prev => prev.map(f => 
            f.id === fileId ? { 
              ...f, 
              status: 'error', 
              error: 'Processing timeout'
            } : f
          ));
        }
      } catch (error) {
        console.error('Status polling failed:', error);
        setFiles(prev => prev.map(f => 
          f.id === fileId ? { 
            ...f, 
            status: 'error', 
            error: 'Status check failed'
          } : f
        ));
      }
    };

    setTimeout(poll, 2000); // Start polling after 2 seconds
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return <Loader className="w-4 h-4 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <File className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusText = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
        return 'Uploading...';
      case 'processing':
        return 'Processing...';
      case 'completed':
        return 'Ready';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="w-full">
      {/* Drop Zone */}
      <div
        className={`
          border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${isDragging 
            ? 'border-blue-400 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Upload Documents
        </h3>
        <p className="text-gray-600 mb-4">
          Drag and drop files here, or click to select
        </p>
        <label className="cursor-pointer">
          <input
            type="file"
            multiple
            accept=".pdf,.doc,.docx,.txt,.md"
            onChange={handleFileSelect}
            className="hidden"
          />
          <span className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50">
            Choose Files
          </span>
        </label>
        <p className="text-xs text-gray-500 mt-2">
          Supports PDF, Word, Text, and Markdown files (max 100MB)
        </p>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-medium text-gray-900 mb-3">
            Uploaded Files ({files.length})
          </h4>
          <div className="space-y-2">
            {files.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  {getStatusIcon(file.status)}
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(file.size)} • {getStatusText(file.status)}
                    </p>
                    {file.error && (
                      <p className="text-xs text-red-600">{file.error}</p>
                    )}
                  </div>
                </div>
                
                {file.status === 'uploading' && (
                  <div className="w-20">
                    <div className="bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all"
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}