import os
import re
import json
from typing import List, Dict, Any

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("[-] Docling not available. Please ensure it's installed via requirements.txt")

def chunk_markdown_text(md_text: str, source_name: str, max_chars: int = 2000) -> List[Dict[Any, Any]]:
    """
    Splits markdown structurally by headers/sections.
    If a section exceeds max_chars, it sub-splits by paragraphs.
    """
    chunks = []
    # Split using markdown headers as delimiters
    sections = re.split(r'(^#+\s+.*)', md_text, flags=re.MULTILINE)
    
    current_section_title = "Introduction/Header"
    
    for item in sections:
        if not item.strip():
            continue
        
        # If this item is a section header, update tracker
        if item.strip().startswith('#'):
            current_section_title = item.strip().lstrip('#').strip()
            continue
        
        text_content = item.strip()
        
        # Split by paragraph if section text is too long
        if len(text_content) > max_chars:
            paragraphs = text_content.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current_chunk) + len(para) <= max_chars:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": {
                                "source": source_name,
                                "section": current_section_title,
                                "type": "pdf_chunk"
                            }
                        })
                    current_chunk = para + "\n\n"
            
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {
                        "source": source_name,
                        "section": current_section_title,
                        "type": "pdf_chunk"
                    }
                })
        else:
            chunks.append({
                "text": text_content,
                "metadata": {
                    "source": source_name,
                    "section": current_section_title,
                    "type": "pdf_chunk"
                }
            })
            
    return chunks

def extract_pdf_directory(pdf_dir: str) -> List[Dict[str, Any]]:
    """
    Iterates through a directory of PDFs, processes them with Docling,
    and returns a combined list of semantic text chunks.
    """
    all_chunks = []
    if not os.path.exists(pdf_dir):
        print(f"[-] Target directory {pdf_dir} does not exist. Skipping PDF parsing.")
        return all_chunks

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"[*] No PDFs found in {pdf_dir}. Placing placeholder structure.")
        return all_chunks

    print(f"[*] Found {len(pdf_files)} PDF(s) to process via Docling...")
    try:
        converter = DocumentConverter()
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"  -> Converting {pdf_file}...")
            try:
                result = converter.convert(pdf_path)
                markdown_text = result.document.export_to_markdown()
                file_chunks = chunk_markdown_text(markdown_text, source_name=pdf_file)
                all_chunks.extend(file_chunks)
                print(f"  [+] Extracted {len(file_chunks)} chunks from {pdf_file}")
            except Exception as e:
                print(f"  [-] Failed to parse document {pdf_file}: {e}. Continuing...")
    except Exception as e:
        print(f"[-] Could not initialize DocumentConverter: {e}")
        
    return all_chunks