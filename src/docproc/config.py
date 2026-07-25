"""Configuration (12-factor: env vars with DOCPROC_ prefix)."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCPROC_", env_file=".env", extra="ignore")

    # LLM provider: "mock" (offline, deterministic) | "openai" | "anthropic"
    llm_provider: str = "mock"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-6"
    max_llm_retries: int = 2          # instructor retry-with-feedback budget

    # OCR
    ocr_lang: str = "eng"
    min_native_text_chars: int = 200  # below this per page => treat PDF as scanned
    use_easyocr: bool = True          # second engine, if installed
    vision_fallback_threshold: float = 0.55  # OCR confidence below this => vision model

    # Routing thresholds (the key business knobs)
    auto_approve_threshold: float = 0.85
    fast_review_threshold: float = 0.60

    # Storage
    db_path: Path = PROJECT_ROOT / "data" / "docproc.db"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # Business rules
    known_vendors: str = Field(
        default="Northwind Supplies GmbH,ACME Industrial AG,Contoso Services Ltd",
        description="Comma-separated vendor allowlist for the business-rule check.",
    )
    max_plausible_invoice_total: float = 100_000.0

    @property
    def vendor_set(self) -> set[str]:
        return {v.strip().lower() for v in self.known_vendors.split(",") if v.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
