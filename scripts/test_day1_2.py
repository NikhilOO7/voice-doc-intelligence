# scripts/test_day1_2.py
import asyncio
import aiohttp
import aiofiles
from pathlib import Path

async def test_document_upload():
    """Test document upload and processing"""
    
    # Create a test document
    test_content = """
# Test Document for Voice Intelligence System

## Introduction
This is a test document to verify the document processing pipeline.

## Section 1: Document Processing
The system should extract this content and create semantic chunks.

### Subsection 1.1: Chunking
Each section should be properly identified and processed.

## Section 2: Embeddings
Multiple embedding models will be used:
- OpenAI text-embedding-3-large
- Voyage-3-large (if available)
- Local fallback model

## Conclusion
The document should be processed successfully with contextual embeddings.
"""
    
    test_file = Path("test_document.md")
    async with aiofiles.open(test_file, "w") as f:
        await f.write(test_content)
    
    # Upload document
    async with aiohttp.ClientSession() as session:
        with open(test_file, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="test_document.md", content_type="text/markdown")
            data.add_field("metadata", '{"category": "test", "author": "system"}')
            
            async with session.post("http://localhost:8000/api/v1/documents/upload", data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Document uploaded: {result['document_id']}")
                    return result['document_id']
                else:
                    print(f"❌ Upload failed: {await resp.text()}")
                    return None

async def check_document_status(document_id: str):
    """Check document processing status"""
    async with aiohttp.ClientSession() as session:
        # Check status every 2 seconds
        for _ in range(30):  # Max 1 minute
            async with session.get(f"http://localhost:8000/api/v1/documents/{document_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status = data['processing_status']
                    print(f"📊 Status: {status}")
                    
                    if status == "completed":
                        print(f"✅ Processing completed!")
                        print(f"  Chunks: {data['metadata'].get('chunks_count', 0)}")
                        print(f"  Embeddings: {data['metadata'].get('embeddings_stored', {})}")
                        return True
                    elif status == "failed":
                        print(f"❌ Processing failed: {data.get('error', 'Unknown error')}")
                        return False
            
            await asyncio.sleep(2)
        
        print("⏱️ Timeout waiting for processing")
        return False

async def test_health_check():
    """Test API health endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/health") as resp:
            if resp.status == 200:
                data = await resp.json()
                print("🏥 Health Check:")
                for service, status in data['services'].items():
                    print(f"  - {service}: {status}")
                return True
            else:
                print("❌ Health check failed")
                return False

async def main():
    print("🧪 Testing Day 1-2 Implementation\n")
    
    # Test health check
    print("1️⃣ Testing health check...")
    await test_health_check()
    print()
    
    # Test document upload
    print("2️⃣ Testing document upload...")
    document_id = await test_document_upload()
    print()
    
    if document_id:
        # Check processing status
        print("3️⃣ Checking document processing...")
        await check_document_status(document_id)

if __name__ == "__main__":
    asyncio.run(main())