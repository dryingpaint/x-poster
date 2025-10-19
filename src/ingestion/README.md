# Ingestion Module

## Overview
Handles document ingestion pipeline: PDF processing, OCR, text extraction, and chunking for vector search.

## Responsibilities
- **PDF Processing**: Extract text and images from PDF documents
- **OCR**: Optical character recognition for scanned documents and images
- **Text Chunking**: Split documents into search-optimized chunks
- **Content Normalization**: Clean and standardize extracted text

## Key Files
- `pdf_processor.py`: PDF parsing and text extraction
- `ocr.py`: OCR using Tesseract/OCRmyPDF
- `chunker.py`: Semantic chunking with overlap

## Development Guidelines

### Working on PDF Processing (`pdf_processor.py`)
**What you can do in parallel:**
- Add support for new document formats (DOCX, PPTX, etc.)
- Improve table extraction logic
- Add image extraction and captioning
- Add metadata extraction (author, title, dates)

**What requires coordination:**
- Changing the Item output format (impacts database operations)
- Modifying text normalization (may affect search quality)
- Changing page splitting logic (impacts chunk boundaries)

**Testing requirements:**
```bash
# Test with sample PDFs
uv run pytest tests/ingestion/test_pdf_processor.py

# Test with various PDF types
uv run pytest tests/ingestion/test_pdf_processor.py::test_scanned_pdf
uv run pytest tests/ingestion/test_pdf_processor.py::test_native_pdf
uv run pytest tests/ingestion/test_pdf_processor.py::test_complex_layout
```

**Code style:**
```python
# GOOD: Return structured data with metadata
from src.core.models import Item, ItemKind

async def process_pdf(file_path: str) -> Item:
    """Process PDF and return Item with extracted text.

    Args:
        file_path: Path to PDF file

    Returns:
        Item with extracted text and metadata

    Raises:
        PDFProcessingError: If PDF is corrupted or unreadable
    """
    # Extract text, images, metadata
    text = extract_text(file_path)
    meta = extract_metadata(file_path)

    return Item(
        kind=ItemKind.PDF,
        title=meta.get("title"),
        source_uri=f"file://{file_path}",
        content_text=text,
        meta=meta,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

# GOOD: Handle different PDF types
def extract_text(file_path: str) -> str:
    """Extract text from PDF, using OCR if needed."""
    doc = fitz.open(file_path)

    # Try native text extraction first
    text = ""
    for page in doc:
        text += page.get_text()

    # Fall back to OCR if little/no text
    if len(text.strip()) < 100:
        text = ocr_pdf(file_path)

    return text
```

### Working on OCR (`ocr.py`)
**What you can do in parallel:**
- Add support for multiple OCR engines (EasyOCR, PaddleOCR)
- Improve OCR quality with preprocessing (deskew, denoise)
- Add language detection for multi-language documents
- Add confidence scoring for OCR results

**What requires coordination:**
- Changing OCR output format (impacts text normalization)
- Modifying system dependencies (Tesseract installation)
- Changing default OCR settings (may affect quality/speed)

**Testing requirements:**
```bash
# Test OCR accuracy
uv run pytest tests/ingestion/test_ocr.py

# Test with various image types
uv run pytest tests/ingestion/test_ocr.py::test_scanned_document
uv run pytest tests/ingestion/test_ocr.py::test_screenshot
uv run pytest tests/ingestion/test_ocr.py::test_poor_quality_image
```

**Code style:**
```python
# GOOD: Return confidence scores with OCR results
from dataclasses import dataclass

@dataclass
class OCRResult:
    text: str
    confidence: float  # 0.0-1.0
    language: str
    bounding_boxes: list[tuple[int, int, int, int]] | None = None

async def ocr_image(image_path: str) -> OCRResult:
    """Run OCR on image and return structured result.

    Args:
        image_path: Path to image file

    Returns:
        OCRResult with extracted text and metadata
    """
    # Preprocess image
    img = preprocess_image(image_path)

    # Run OCR
    result = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Calculate confidence
    confidences = [int(c) for c in result['conf'] if c != '-1']
    avg_confidence = sum(confidences) / len(confidences) / 100.0

    return OCRResult(
        text=" ".join(result['text']),
        confidence=avg_confidence,
        language=result.get('lang', 'eng')
    )

# GOOD: Optimize images before OCR
def preprocess_image(image_path: str) -> Image:
    """Preprocess image to improve OCR accuracy."""
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert('L')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Resize if too small
    if img.width < 1000:
        img = img.resize((img.width * 2, img.height * 2))

    return img
```

### Working on Chunking (`chunker.py`)
**What you can do in parallel:**
- Add semantic chunking strategies (sentence boundaries, paragraphs)
- Add chunk size optimization based on content type
- Add metadata preservation in chunks (section headings, page numbers)
- Add chunk quality scoring

**What requires coordination:**
- Changing chunk size/overlap (affects embedding quality and search)
- Modifying ItemChunk output format (impacts database schema)
- Changing chunk ID generation (may affect deduplication)

**Testing requirements:**
```bash
# Test chunking strategies
uv run pytest tests/ingestion/test_chunker.py

# Test chunk quality
uv run pytest tests/ingestion/test_chunker.py::test_chunk_coherence
uv run pytest tests/ingestion/test_chunker.py::test_chunk_size_distribution
```

**Code style:**
```python
# GOOD: Configurable chunking with sensible defaults
from src.core.models import Item, ItemChunk

def chunk_item(
    item: Item,
    chunk_size: int = 512,  # tokens
    chunk_overlap: int = 50,  # tokens
    preserve_sentences: bool = True
) -> list[ItemChunk]:
    """Split item into overlapping chunks for retrieval.

    Args:
        item: Item to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        preserve_sentences: Don't split mid-sentence

    Returns:
        List of ItemChunk objects
    """
    chunks = []
    text = item.content_text

    # Split into sentences if preserving boundaries
    if preserve_sentences:
        sentences = split_sentences(text)
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            if current_tokens + sentence_tokens > chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(ItemChunk(
                    item_id=item.item_id,
                    content=chunk_text
                ))

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-3:]  # Keep last 3 sentences
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(ItemChunk(
                item_id=item.item_id,
                content=" ".join(current_chunk)
            ))

    return chunks

# GOOD: Add chunk metadata for debugging
def chunk_with_metadata(item: Item) -> list[ItemChunk]:
    """Chunk item and preserve source metadata."""
    chunks = []

    for i, chunk_text in enumerate(split_text(item.content_text)):
        chunk = ItemChunk(
            item_id=item.item_id,
            content=chunk_text,
        )
        # Add metadata through Item.meta if needed
        chunks.append(chunk)

    return chunks
```

## Interface Contract

### PDF Processing
```python
from src.ingestion.pdf_processor import process_pdf

# Process single PDF
item = await process_pdf("path/to/document.pdf")
# Returns: Item with extracted text and metadata
```

### OCR
```python
from src.ingestion.ocr import ocr_image, ocr_pdf

# OCR single image
result = await ocr_image("path/to/scan.png")
# Returns: OCRResult with text and confidence

# OCR PDF (for scanned documents)
text = await ocr_pdf("path/to/scanned.pdf")
# Returns: str with extracted text
```

### Chunking
```python
from src.ingestion.chunker import chunk_item

# Chunk document for retrieval
chunks = chunk_item(
    item=item,
    chunk_size=512,
    chunk_overlap=50
)
# Returns: list[ItemChunk]
```

## Pipeline Flow
```
PDF/Image → Extract Text → Normalize → Chunk → Store
              ↓
         OCR if needed
```

### Full Ingestion Example
```python
from src.ingestion.pdf_processor import process_pdf
from src.ingestion.chunker import chunk_item
from src.db.operations import insert_item, insert_chunks

# 1. Process PDF
item = await process_pdf("document.pdf")

# 2. Chunk for retrieval
chunks = chunk_item(item, chunk_size=512)

# 3. Store in database
await insert_item(item)
await insert_chunks(chunks)
```

## Dependencies
**External:**
- `PyMuPDF` (fitz): PDF parsing and text extraction
- `pytesseract`: OCR interface
- `Pillow`: Image processing
- `ocrmypdf`: PDF OCR preprocessing
- `tiktoken`: Token counting for chunking

**Internal:**
- `src.core.models`: Item, ItemChunk, ItemKind
- `src.core.config`: Ingestion settings

## Performance Considerations
- **PDF Processing**: ~1-5s per document (native text)
- **OCR**: ~10-30s per page (depends on image quality)
- **Chunking**: ~100ms per document
- **Parallelization**: Process multiple documents concurrently
- **Memory**: Load large PDFs in streaming mode

### Optimization Tips
```python
# GOOD: Process PDFs in parallel
import asyncio
items = await asyncio.gather(*[
    process_pdf(pdf_path)
    for pdf_path in pdf_files
])

# GOOD: Use streaming for large PDFs
def process_large_pdf(file_path: str, max_pages: int = 1000):
    """Process PDF in chunks to avoid memory issues."""
    doc = fitz.open(file_path)

    for i in range(0, len(doc), max_pages):
        batch = doc[i:i+max_pages]
        yield process_pdf_batch(batch)

# GOOD: Cache OCR results
from functools import lru_cache

@lru_cache(maxsize=100)
def ocr_image_cached(image_path: str) -> str:
    """Cache OCR results for repeated images."""
    return ocr_image(image_path).text
```

## Common Pitfalls
- **DON'T** load entire PDF into memory (use streaming)
- **DON'T** skip OCR check for "native" PDFs (some have hidden images)
- **DON'T** chunk mid-sentence (hurts search quality)
- **DON'T** forget to normalize whitespace and encoding
- **DO** handle corrupted/encrypted PDFs gracefully
- **DO** preserve metadata (page numbers, sections, etc.)
- **DO** validate chunk sizes (not too small/large)

## Testing Checklist
- [ ] PDF processing handles native and scanned PDFs
- [ ] OCR quality > 95% accuracy on clean images
- [ ] Chunking preserves sentence boundaries
- [ ] Chunks are within target size (512 ± 50 tokens)
- [ ] Metadata extraction works for various PDF types
- [ ] Error handling for corrupted files
- [ ] Performance: <5s per PDF, <30s per OCR page

## Local Development Setup
```bash
# Install system dependencies (macOS)
brew install tesseract ocrmypdf

# Install system dependencies (Ubuntu)
sudo apt-get install tesseract-ocr ocrmypdf

# Install Python dependencies
uv sync

# Run tests with sample documents
uv run pytest tests/ingestion/ -v
```

## Contact for Coordination
When modifying ingestion pipeline:
1. Test with diverse document samples (PDFs, scans, etc.)
2. Measure impact on search quality (Precision@10)
3. Benchmark processing time (should stay <5s per PDF)
4. Coordinate with database module on schema changes
