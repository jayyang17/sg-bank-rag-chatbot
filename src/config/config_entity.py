from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathConfig:
    raw_dir: Path
    processed_dir: Path
    output_dir: Path
    
@dataclass
class RetrievalConfig:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    vector_store_path: str

@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int

@dataclass
class PromptConfig:
    prompt_template: str

@dataclass
class TelegramConfig:
    token: str

@dataclass
class FlaskConfig:
    port: int

@dataclass
class AWSConfig:
    s3_bucket: str
