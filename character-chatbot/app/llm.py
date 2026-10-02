import os
from anthropic import Anthropic

if not os.getenv("ANTHROPIC_API_KEY"):
    print("[llm] ANTHROPIC_API_KEY가 설정되지 않았습니다 — LLM 호출이 실패합니다.")

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODELS = {
    "extraction": "claude-haiku-4-5-20251001",  # 저렴/빠름, 파일 하나당 한 번씩 실행
    "roleplay": "claude-sonnet-5",  # 실제 캐릭터 응답용, 품질 우선
}


def extract_text(response) -> str:
    """Anthropic responses are a list of content blocks; pull out the text block."""
    for block in response.content:
        if block.type == "text":
            return block.text
    raise RuntimeError(
        f"Model response contained no text block (stop_reason={getattr(response, 'stop_reason', '?')}, "
        f"content_block_types={[getattr(b, 'type', '?') for b in response.content]})"
    )