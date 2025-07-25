// apps/web/src/components/DocumentList.tsx
import React, { useState, useEffect } from 'react';
import { File, Search, Calendar, Download, Trash2, Eye, Loader } from 'lucide-react';

interface Document {
  id: string;
  external_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  processing_status: string;
  created_at: string;
  metadata?: {
    chunks_count?: number;
    embeddings_stored?: Record<string, number>;
  };
}

interface DocumentListResponse {
  documents: Document[];
  pagination: {
    total: number;
    skip: number;
    limit: number;
    pages: number;
  };
}

export default function DocumentList() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDocuments();
  }, [currentPage, statusFilter]);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams({
        skip: ((currentPage - 1) * 10).toString(),
        limit: '10',
      });

      if (statusFilter !== 'all') {
        params.append('status', statusFilter);
      }

      const response = await fetch(`/api/v1/documents?${params}`);
      if (!response.ok) {
        throw new Error('Failed to fetch documents');
      }

      const data: DocumentListResponse = await response.json();
      setDocuments(data.documents);
      setTotalPages(data.pagination.pages);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (documentId: string) => {
    if (!confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`/api/v1/documents/${documentId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete document');
      }

      // Remove from local state
      setDocuments(prev => prev.filter(doc => doc.id !== documentId));
      
      // Refresh the list to get accurate counts
      fetchDocuments();
    } catch (err) {
      console.error('Error deleting document:', err);
      alert('Failed to delete document. Please try again.');
    }
  };

  const handleViewDetails = async (document: Document) => {
    try {
      const response = await fetch(`/api/v1/documents/${document.id}?include_chunks=true`);
      if (!response.ok) {
        throw new Error('Failed to fetch document details');
      }

      const details = await response.json();
      setSelectedDocument({ ...document, ...details });
    } catch (err) {
      console.error('Error fetching document details:', err);
      alert('Failed to load document details');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      pending: { color: 'bg-yellow-100 text-yellow-800', label: 'Pending' },
      queued: { color: 'bg-blue-100 text-blue-800', label: 'Queued' },
      processing: { color: 'bg-purple-100 text-purple-800', label: 'Processing' },
      completed: { color: 'bg-green-100 text-green-800', label: 'Ready' },
      failed: { color: 'bg-red-100 text-red-800', label: 'Failed' },
    };

    const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.pending;
    
    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.color}`}>
        {config.label}
      </span>
    );
  };

  const filteredDocuments = documents.filter(doc =>
    doc.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader className="w-6 h-6 animate-spin text-gray-500" />
        <span className="ml-2 text-gray-600">Loading documents...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <div className="text-red-600 mb-2">⚠️ Error loading documents</div>
        <p className="text-gray-600 mb-4">{error}</p>
        <button 
          onClick={fetchDocuments}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Search and Filter Controls */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="all">All Status</option>
          <option value="completed">Ready</option>
          <option value="processing">Processing</option>
          <option value="failed">Failed</option>
          <option value="pending">Pending</option>
        </select>
      </div>

      {/* Document List */}
      {filteredDocuments.length === 0 ? (
        <div className="text-center py-8">
          <File className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
          <p className="text-gray-600">
            {searchTerm ? 'Try adjusting your search criteria' : 'Upload your first document to get started'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredDocuments.map((document) => (
            <div
              key={document.id}
              className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  <File className="w-6 h-6 text-gray-400 mt-1" />
                  
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-gray-900 truncate">
                      {document.filename}
                    </h3>
                    
                    <div className="flex items-center space-x-4 mt-1 text-xs text-gray-500">
                      <span>{formatFileSize(document.file_size)}</span>
                      <span>•</span>
                      <span className="flex items-center">
                        <Calendar className="w-3 h-3 mr-1" />
                        {formatDate(document.created_at)}
                      </span>
                      {document.metadata?.chunks_count && (
                        <>
                          <span>•</span>
                          <span>{document.metadata.chunks_count} chunks</span>
                        </>
                      )}
                    </div>
                    
                    <div className="mt-2">
                      {getStatusBadge(document.processing_status)}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2 ml-4">
                  <button
                    onClick={() => handleViewDetails(document)}
                    className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
                    title="View details"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={() => handleDelete(document.id)}
                    className="p-2 text-gray-400 hover:text-red-600 rounded-lg hover:bg-gray-100"
                    title="Delete document"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center space-x-2 mt-6">
          <button
            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
            disabled={currentPage === 1}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Previous
          </button>
          
          <span className="px-4 py-2 text-sm text-gray-600">
            Page {currentPage} of {totalPages}
          </span>
          
          <button
            onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
            disabled={currentPage === totalPages}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      )}

      {/* Document Details Modal */}
      {selectedDocument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h2 className="text-lg font-semibold text-gray-900">
                  Document Details
                </h2>
                <button
                  onClick={() => setSelectedDocument(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Filename</label>
                  <p className="text-sm text-gray-900">{selectedDocument.filename}</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">File Size</label>
                    <p className="text-sm text-gray-900">{formatFileSize(selectedDocument.file_size)}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Type</label>
                    <p className="text-sm text-gray-900">{selectedDocument.file_type}</p>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700">Status</label>
                  <div className="mt-1">
                    {getStatusBadge(selectedDocument.processing_status)}
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700">Upload Date</label>
                  <p className="text-sm text-gray-900">{formatDate(selectedDocument.created_at)}</p>
                </div>
                
                {selectedDocument.metadata && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Processing Info</label>
                    <div className="mt-1 text-sm text-gray-900">
                      {selectedDocument.metadata.chunks_count && (
                        <p>Chunks: {selectedDocument.metadata.chunks_count}</p>
                      )}
                      {selectedDocument.metadata.embeddings_stored && (
                        <p>Embeddings: {Object.values(selectedDocument.metadata.embeddings_stored).reduce((a, b) => a + b, 0)} vectors</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}