'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, AlertCircle } from 'lucide-react';
import { apiClient } from '@/lib/api-client';
import { formatBytes } from '@/lib/utils';

interface DocumentUploadProps {
  onUploadSuccess?: (documentId: string) => void;
  onUploadError?: (error: string) => void;
}

export default function DocumentUpload({ onUploadSuccess, onUploadError }: DocumentUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [useEnhanced, setUseEnhanced] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false,
  });

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setError(null);

    try {
      const result = await apiClient.uploadDocument(selectedFile, useEnhanced);
      console.log('Upload successful:', result);

      if (onUploadSuccess) {
        onUploadSuccess(result.document_id);
      }

      // Reset state
      setSelectedFile(null);
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || err.message || 'Upload failed';
      setError(errorMessage);

      if (onUploadError) {
        onUploadError(errorMessage);
      }
    } finally {
      setUploading(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setError(null);
  };

  return (
    <div className="w-full space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-all duration-200
          ${isDragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400 bg-white'
          }
          ${selectedFile ? 'bg-gray-50' : ''}
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center gap-4">
          <div className={`
            p-4 rounded-full
            ${isDragActive ? 'bg-primary-100' : 'bg-gray-100'}
          `}>
            <Upload className={`
              w-8 h-8
              ${isDragActive ? 'text-primary-600' : 'text-gray-500'}
            `} />
          </div>

          <div>
            <p className="text-lg font-medium text-gray-900">
              {isDragActive ? 'Drop your document here' : 'Upload a document'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Drag and drop or click to browse
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Supports: PDF, DOCX, DOC, TXT (max 100MB)
            </p>
          </div>
        </div>
      </div>

      {/* Selected File */}
      {selectedFile && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3 flex-1">
              <div className="p-2 bg-primary-50 rounded">
                <FileText className="w-5 h-5 text-primary-600" />
              </div>

              <div className="flex-1 min-w-0">
                <p className="font-medium text-gray-900 truncate">
                  {selectedFile.name}
                </p>
                <p className="text-sm text-gray-500">
                  {formatBytes(selectedFile.size)}
                </p>
              </div>
            </div>

            <button
              onClick={clearSelection}
              className="p-1 hover:bg-gray-100 rounded transition-colors"
              disabled={uploading}
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {/* Enhanced Processing Option */}
          <div className="mt-4 pt-4 border-t border-gray-100">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useEnhanced}
                onChange={(e) => setUseEnhanced(e.target.checked)}
                disabled={uploading}
                className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">
                Use enhanced processing
                <span className="text-xs text-gray-500 ml-1">
                  (3-level contextual embeddings)
                </span>
              </span>
            </label>
          </div>

          {/* Upload Button */}
          <div className="mt-4">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className={`
                w-full py-2.5 px-4 rounded-lg font-medium
                transition-all duration-200
                ${uploading
                  ? 'bg-gray-300 cursor-not-allowed text-gray-600'
                  : 'bg-primary-600 hover:bg-primary-700 text-white shadow-sm hover:shadow'
                }
              `}
            >
              {uploading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Uploading...
                </span>
              ) : (
                'Upload Document'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-red-900">Upload Failed</p>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
