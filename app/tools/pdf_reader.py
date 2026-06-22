import base64
from io import BytesIO

import litellm
from pdf2image import convert_from_path

from app.config import settings
from app.debug_log import log_step

VISION_PROMPT = (
    "Transcribe all text from this resume image exactly as written. "
    "Then on a new line starting with 'STRUCTURE:', note any multi-column "
    "layout, tables, or text in headers/footers that could confuse an ATS parser."
)


def encode_image(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def transcribe_page(image) -> str:
    encoded = encode_image(image)
    response = litellm.completion(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"},
            ],
        }],
        max_tokens=settings.max_output_tokens,
    )
    text = response.choices[0].message.content
    if not text.strip():
        log_step("Vision OCR returned empty text for a page, likely truncated by the output token limit")
    return text


def read_pdf(file_path: str) -> str:
    pages = convert_from_path(file_path)
    return "\n\n".join(transcribe_page(page) for page in pages)
