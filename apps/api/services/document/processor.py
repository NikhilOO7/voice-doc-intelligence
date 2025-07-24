# apps/api/services/document/processor.py
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
import pypdf
import docx
from unstructured.partition.auto import partition
import nltk
from sentence_transformers import SentenceTransformer

from ...core.config import settings

logger = logging.getLogger(__name__)

class ModernDocumentProcessor:
    """Advanced document processor using IBM Docling and modern techniques"""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        # Initialize Docling for advanced PDF processing
        self.pdf_pipeline_options = PdfPipelineOptions()
        self.pdf_pipeline_options.do_ocr = True
        self.pdf_pipeline_options.do_table_structure = True
        
        # Initialize sentence splitter
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    async def process_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Process document with advanced structure extraction"""
        logger.info(f"Processing document: {file_path} (type: {file_type})")
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(file_path)
        
        # Process based on file type
        if file_type.startswith("application/pdf"):
            result = await self._process_pdf_advanced(file_path)
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            result = await self._process_docx_advanced(file_path)
        else:
            result = await self._process_with_unstructured(file_path)
        
        # Add file hash
        result["file_hash"] = file_hash
        
        # Create semantic chunks
        result["chunks"] = await self._create_semantic_chunks(result)
        
        return result
    
    async def _process_pdf_advanced(self, file_path: str) -> Dict[str, Any]:
        """Process PDF using Docling for better structure extraction"""
        try:
            # Use Docling for advanced processing
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                pdf_pipeline=StandardPdfPipeline(pipeline_options=self.pdf_pipeline_options)
            )
            
            doc_result = converter.convert(file_path)
            
            # Extract structured content
            content = {
                "text": doc_result.document.export_to_markdown(),
                "pages": [],
                "structure": {
                    "headings": [],
                    "tables": [],
                    "lists": [],
                    "figures": []
                },
                "metadata": doc_result.document.metadata.dict() if doc_result.document.metadata else {}
            }
            
            # Process pages and structure
            for page in doc_result.document.pages:
                page_data = {
                    "page_number": page.page_number,
                    "text": page.text,
                    "tables": [table.dict() for table in page.tables] if hasattr(page, 'tables') else [],
                    "figures": [fig.dict() for fig in page.figures] if hasattr(page, 'figures') else []
                }
                content["pages"].append(page_data)
                
                # Extract structure elements
                if page.tables:
                    content["structure"]["tables"].extend(page.tables)
                if hasattr(page, 'figures') and page.figures:
                    content["structure"]["figures"].extend(page.figures)
            
            # Extract headings from markdown
            lines = content["text"].split('\n')
            for line in lines:
                if line.startswith('#'):
                    level = len(line.split()[0])
                    content["structure"]["headings"].append({
                        "level": level,
                        "text": line.strip('#').strip()
                    })
            
            return content
            
        except Exception as e:
            logger.warning(f"Docling processing failed, falling back: {e}")
            return await self._process_pdf_fallback(file_path)
    
    async def _process_pdf_fallback(self, file_path: str) -> Dict[str, Any]:
        """Fallback PDF processing"""
        content = {
            "text": "",
            "pages": [],
            "structure": {"headings": [], "tables": []}
        }
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                content["pages"].append({
                    "page_number": page_num + 1,
                    "text": text
                })
                content["text"] += text + "\n"
        
        return content
    
    async def _process_docx_advanced(self, file_path: str) -> Dict[str, Any]:
        """Process Word documents with structure preservation"""
        doc = docx.Document(file_path)
        content = {
            "text": "",
            "paragraphs": [],
            "structure": {
                "headings": [],
                "lists": [],
                "tables": []
            }
        }
        
        current_list = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                para_data = {
                    "text": para.text,
                    "style": para.style.name,
                    "level": para.style.priority if para.style.priority else None,
                    "is_list": para.style.name.startswith('List')
                }
                
                content["paragraphs"].append(para_data)
                content["text"] += para.text + "\n"
                
                # Track headings
                if "Heading" in para.style.name:
                    content["structure"]["headings"].append({
                        "level": int(para.style.name[-1]) if para.style.name[-1].isdigit() else 1,
                        "text": para.text
                    })
                
                # Track lists
                if para_data["is_list"]:
                    current_list.append(para.text)
                elif current_list:
                    content["structure"]["lists"].append(current_list)
                    current_list = []
        
        # Process tables
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            content["structure"]["tables"].append({
                "data": table_data,
                "rows": len(table.rows),
                "cols": len(table.columns)
            })
        
        return content
    
    async def _process_with_unstructured(self, file_path: str) -> Dict[str, Any]:
        """Process other document types using unstructured"""
        elements = partition(filename=file_path)
        
        content = {
            "text": "",
            "elements": [],
            "structure": {"headings": [], "tables": [], "lists": []}
        }
        
        for element in elements:
            element_data = {
                "type": element.category,
                "text": str(element),
                "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {}
            }
            content["elements"].append(element_data)
            content["text"] += str(element) + "\n"
            
            # Extract structure based on element type
            if element.category == "Title":
                content["structure"]["headings"].append({
                    "level": 1,
                    "text": str(element)
                })
        
        return content
    
    async def _create_semantic_chunks(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks that preserve meaning and context"""
        chunks = []
        
        # If we have structured content, use it
        if "paragraphs" in content:
            chunks = await self._chunk_by_structure(content)
        elif "pages" in content:
            chunks = await self._chunk_by_pages(content)
        else:
            chunks = await self._chunk_by_sentences(content["text"])
        
        # Add chunk metadata
        for idx, chunk in enumerate(chunks):
            chunk["chunk_index"] = idx
            chunk["token_count"] = len(chunk["content"].split())
            
            # Add structural context
            if "structure" in content:
                chunk["structure_context"] = self._get_structure_context(
                    chunk, content["structure"]
                )
        
        return chunks
    
    async def _chunk_by_structure(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk by document structure (paragraphs, sections)"""
        chunks = []
        current_chunk = ""
        current_metadata = {}
        current_section = []
        
        for para in content["paragraphs"]:
            # Check if this is a heading
            if "Heading" in para.get("style", ""):
                # Save current chunk if exists
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": current_metadata,
                        "section_path": current_section.copy()
                    })
                
                # Update section path
                level = int(para["style"][-1]) if para["style"][-1].isdigit() else 1
                current_section = current_section[:level-1] + [para["text"]]
                
                # Start new chunk with heading
                current_chunk = para["text"] + "\n"
                current_metadata = {"style": para["style"], "is_heading": True}
            
            elif len(current_chunk) + len(para["text"]) > self.chunk_size:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": current_metadata,
                        "section_path": current_section.copy()
                    })
                
                # Start new chunk
                current_chunk = para["text"]
                current_metadata = {"style": para.get("style", "Normal")}
            
            else:
                # Add to current chunk
                current_chunk += "\n" + para["text"]
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": current_metadata,
                "section_path": current_section.copy()
            })
        
        return chunks
    
    async def _chunk_by_pages(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk by pages with overlap"""
        chunks = []
        
        for page_data in content["pages"]:
            page_text = page_data["text"]
            page_num = page_data["page_number"]
            
            # Split page into sentences
            try:
                sentences = nltk.sent_tokenize(page_text)
            except:
                sentences = page_text.split('. ')
            
            # Create chunks from sentences
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) > self.chunk_size:
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "metadata": {
                                "start_page": page_num,
                                "end_page": page_num
                            }
                        })
                    current_chunk = sent
                else:
                    current_chunk += " " + sent
            
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        "start_page": page_num,
                        "end_page": page_num
                    }
                })
        
        return chunks
    
    async def _chunk_by_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Fallback: chunk by sentences with sliding window"""
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split('. ')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            sent_size = len(sent)
            
            if current_size + sent_size > self.chunk_size and current_chunk:
                # Create chunk
                chunks.append({
                    "content": ' '.join(current_chunk),
                    "metadata": {}
                })
                
                # Overlap: keep last few sentences
                overlap_size = 0
                overlap_sents = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) < self.chunk_overlap:
                        overlap_sents.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sents + [sent]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sent)
                current_size += sent_size
        
        if current_chunk:
            chunks.append({
                "content": ' '.join(current_chunk),
                "metadata": {}
            })
        
        return chunks
    
    def _get_structure_context(self, chunk: Dict[str, Any], structure: Dict[str, Any]) -> Dict[str, Any]:
        """Get structural context for a chunk"""
        context = {
            "nearest_heading": None,
            "in_table": False,
            "in_list": False
        }
        
        # Find nearest heading
        chunk_text = chunk["content"].lower()
        for heading in structure.get("headings", []):
            if heading["text"].lower() in chunk_text:
                context["nearest_heading"] = heading
                break
        
        return context
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()