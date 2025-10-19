"""Script to create test PDF fixtures."""

from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    print("reportlab not installed. Install with: uv add --dev reportlab")
    exit(1)


def create_simple_pdf(output_path: Path) -> None:
    """Create a simple PDF with text content."""
    c = canvas.Canvas(str(output_path), pagesize=letter)

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Test Document: Simple PDF")

    # Add some paragraphs
    c.setFont("Helvetica", 12)
    y = 700
    paragraphs = [
        "This is a test PDF document created for testing the ingestion pipeline.",
        "It contains multiple paragraphs of text that should be extracted correctly.",
        "",
        "The document processing system should be able to:",
        "1. Extract text from native PDF files",
        "2. Parse metadata like title and author",
        "3. Handle multiple pages correctly",
        "4. Preserve sentence boundaries for chunking",
        "",
        "This paragraph contains enough text to test chunking algorithms. " * 5,
    ]

    for para in paragraphs:
        if not para:
            y -= 15
            continue
        # Wrap long text
        words = para.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if c.stringWidth(test_line, "Helvetica", 12) < 400:
                line = test_line
            else:
                c.drawString(100, y, line)
                y -= 15
                line = word + " "
        if line:
            c.drawString(100, y, line)
            y -= 15

    # Add metadata
    c.setAuthor("Test Author")
    c.setTitle("Test Document: Simple PDF")
    c.setSubject("Testing PDF Processing")
    c.setCreator("Test Fixture Generator")

    c.save()


def create_multipage_pdf(output_path: Path) -> None:
    """Create a multi-page PDF with text content."""
    c = canvas.Canvas(str(output_path), pagesize=letter)

    for page_num in range(1, 4):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, f"Test Document: Page {page_num}")

        c.setFont("Helvetica", 12)
        y = 700
        for i in range(20):
            text = f"Page {page_num}, Line {i+1}: This is test content with enough text to test extraction. "
            c.drawString(100, y, text)
            y -= 20
            if y < 100:
                break

        if page_num < 3:
            c.showPage()

    c.setAuthor("Test Author")
    c.setTitle("Test Document: Multi-page")
    c.save()


def create_empty_pdf(output_path: Path) -> None:
    """Create a PDF with minimal/no text (simulates scanned doc)."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    # Just create a blank page
    c.save()


def main():
    """Create all test fixtures."""
    fixtures_dir = Path(__file__).parent

    print("Creating test PDF fixtures...")

    create_simple_pdf(fixtures_dir / "simple.pdf")
    print("✓ Created simple.pdf")

    create_multipage_pdf(fixtures_dir / "multipage.pdf")
    print("✓ Created multipage.pdf")

    create_empty_pdf(fixtures_dir / "empty.pdf")
    print("✓ Created empty.pdf (simulates scanned document)")

    print("\nAll test fixtures created successfully!")


if __name__ == "__main__":
    main()
