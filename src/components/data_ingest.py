import os
import json
import re
import PyPDF2
import pandas as pd
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import camelot

from src.config.config_manager import ConfigurationManager
from src.logging.logger import logging


# ---- SECTION DETECTION LOGIC ---- #
SECTION_KEYWORDS = {
    "Chairman": "Chairman Statement",
    "CEO": "CEO Message",
    "Financial Statements": "Financial Statements",
    "Income Statement": "Financial Statements",
    "Sustainability": "Sustainability",
    "Risk Management": "Risk Management",
    "Corporate Governance": "Corporate Governance",
    "Performance Summary": "Performance Summary",
    "Overview": "Overview",
}

class DataIngestion:
    def __init__(self, cfg: ConfigurationManager):
        self.cfg = cfg
        # path config
        path_config = cfg.get_path_config()
        self.pdf_dir = path_config.raw_dir       
        self.output_json_dir = path_config.output_dir
        os.makedirs(self.output_json_dir, exist_ok=True)

        # retrieval_config
        retrieval_config = cfg.get_retrieval_config()
        self.chunk_size = retrieval_config.chunk_size
        self.chunk_overlap = retrieval_config.chunk_overlap

    def detect_section(self, page_text: str) -> str:
        for keyword, section in SECTION_KEYWORDS.items():
            if re.search(rf"\b{re.escape(keyword)}\b", page_text, re.IGNORECASE):
                return section
        return ""

    # ---- EXTRACT FUNCTIONS ---- #

    def extract_text_by_page(self, pdf_path) -> List[str]:
        """Extract text from each page using PyPDF2."""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return [page.extract_text() or "" for page in reader.pages]


    def extract_tables_by_page(self, pdf_path) -> Dict[int, List[pd.DataFrame]]:
        """Extract tables per page using Camelot."""
        tables_by_page = {}
        total_pages = len(PyPDF2.PdfReader(open(pdf_path, 'rb')).pages)

        for page in range(1, total_pages + 1):
            try:
                tables = camelot.read_pdf(pdf_path, pages=str(page), flavor="lattice")  # or flavor="stream"
                if tables:
                    dfs = [t.df for t in tables]
                    tables_by_page[page] = dfs
            except Exception as e:
                print(f"Camelot failed on page {page}: {e}")
        return tables_by_page


    # ---- TEXT CHUNKING ---- #

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)


    # ---- TABLE TO TEXT ---- #

    def convert_table_to_text(self, df: pd.DataFrame) -> str:
        """Convert table data to a clean text string."""
        rows = df.fillna("").astype(str).values.tolist()
        text_lines = [" | ".join(row) for row in rows if any(cell.strip() for cell in row)]
        return "\n".join(text_lines)


    # ---- ENTRY CREATION ---- #

    def create_jsonl_entries(
        self,
        text_pages: List[str],
        table_pages: Dict[int, List[pd.DataFrame]],
        pdf_name: str,
        bank_name: str,
        report_year: int
    ) -> List[Dict]:
        all_entries = []

        for page_num, page_text in enumerate(text_pages, start=1):
            section = self.detect_section(page_text)
            source = os.path.basename(pdf_name)

            # Text chunks
            for chunk_idx, chunk in enumerate(self.chunk_text(page_text), start=1):
                chunk = chunk.strip()
                if len(chunk) < 50:
                    continue  # filter out short or noisy chunks

                entry_id = f"{bank_name.lower()}-{report_year}-pg{page_num:03d}-chunk{chunk_idx}"
                all_entries.append({
                    "id": entry_id,
                    "text": chunk,
                    "bank": bank_name,
                    "year": report_year,
                    "section": section,
                    "page_number": page_num,
                    "table_data": [],
                    "source": source
                })

            # Table chunks (as text)
            for tbl_idx, df in enumerate(table_pages.get(page_num, []), start=1):
                table_data = df.fillna("").astype(str).to_dict(orient="records")
                table_text = self.convert_table_to_text(df)
                if len(table_text) < 50:
                    continue

                entry_id = f"{bank_name.lower()}-{report_year}-pg{page_num:03d}-table{tbl_idx}"
                all_entries.append({
                    "id": entry_id,
                    "text": table_text,
                    "bank": bank_name,
                    "year": report_year,
                    "section": section,
                    "page_number": page_num,
                    "table_data": table_data,
                    "source": source
                })

        return all_entries


    def write_jsonl(entries: List[Dict], output_path: str):
        """Write the list of entries as JSON Lines to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


    def parse_filename(self, filename: str):
        match = re.match(r"(?P<bank>\w+)-annual-report-(?P<year>\d{4})\.pdf", filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        return match.group("bank"), int(match.group("year"))

    # ---- MAIN FUNCTION ---- #
    def process_all_pdfs(self):
        for file in os.listdir(self.pdf_dir):
            if not file.endswith(".pdf"):
                continue

            pdf_path = os.path.join(self.pdf_dir, file)
            bank, year = self.parse_filename(file)
            jsonl_path = os.path.join(self.output_json_dir, file.replace(".pdf", ".jsonl"))

            logging.info(f"Processing {file}...")

            text_pages = self.extract_text_by_page(pdf_path)
            table_pages = self.extract_tables_by_page(pdf_path)

            entries = self.create_jsonl_entries(text_pages, table_pages, file, bank, year)
            self.write_jsonl(entries, jsonl_path)

            logging.info(f"Saved {len(entries)} chunks to {jsonl_path}")


if __name__ == "__main__":
    cfg = ConfigurationManager()
    data_ingestion = DataIngestion(cfg)
    data_ingestion.process_all_pdfs()
