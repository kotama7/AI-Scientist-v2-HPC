"""PDF utilities for the review module."""

import json

from pypdf import PdfReader
import pymupdf
import pymupdf4llm


def load_paper(pdf_path: str, num_pages: int = None, min_size: int = 100) -> str:
    """Load and extract text from a PDF paper.

    Tries multiple methods in order: pymupdf4llm, pymupdf, pypdf.

    Args:
        pdf_path: Path to the PDF file.
        num_pages: Optional limit on number of pages to extract.
        min_size: Minimum text size to consider valid.

    Returns:
        Extracted text content from the PDF.

    Raises:
        Exception: If text extraction fails from all methods.
    """
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:
                text += page.get_text()
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                pages = reader.pages
            else:
                pages = reader.pages[:num_pages]
            text = "".join(page.extract_text() for page in pages)
            if len(text) < min_size:
                raise Exception("Text too short")
    return text


def load_review(json_path: str) -> dict:
    """Load a review from a JSON file.

    Args:
        json_path: Path to the JSON file containing the review.

    Returns:
        Review dictionary from the loaded JSON.
    """
    with open(json_path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]
