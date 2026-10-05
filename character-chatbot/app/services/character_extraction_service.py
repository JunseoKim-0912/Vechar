from sqlalchemy.orm import Session
from ..llm import generate_structured
from ..schemas import CharacterProfileData

FOCUSED_EXTRACTION_SYSTEM_PROMPT = """당신은 캐릭터 분석가입니다. 여러 개의 텍스트가 주어지고, 그 중 특정 인물 한 명에
대한 정보만 추출해야 합니다. 다른 인물의 정보는 무시하세요. 지목된 인물이 직접 말하거나 행동하거나,
다른 인물이 그 인물에 대해 언급하는 부분만 근거로 삼으세요.

제공된 Structured Outputs schema의 모든 필드를 채우세요. typed timeline에는 이 인물에게 실제로 일어난 사건만
발생 순서의 증거와 함께 기록하세요. 날짜를 지어내지 말고 모호한 시점은 uncertainty로 남기세요.
비수치 이전/이후는 relative_order로 보존하고, 사후 사건은 death event의 이후로 연결하세요.
chat_reference_point는 서버가 timeline에서 계산하므로 null로 반환하세요.
Do not treat narrative order as chronological order. Separate flashbacks, recollections, dreams, forecasts,
and current-time events. A flashback age/state must not replace the latest living canonical state.

지목된 인물에 대한 근거를 텍스트에서 찾을 수 없는 필드는 빈 문자열이나 빈 배열로 두세요."""

FOCUSED_EXTRACTION_SYSTEM_PROMPT += """

Language policy:
- Understand English, Korean, and mixed English/Korean text, including facts split across languages.
- This creates a new canonical profile, so use the source material's natural dominant language and style while incorporating evidence from every language present.
- Preserve proper names and fictional terms as written. Do not translate names merely to match the output language.
- Keep sample_dialogues in their original quotation language."""


def extract_character_from_world_text(
    db: Session, user_id: str, character_name: str, source_texts: list[str]
) -> CharacterProfileData:
    combined = "\n\n---\n\n".join(source_texts)
    return generate_structured(
        db=db,
        user_id=user_id,
        request_type="world_character_extraction",
        task="analysis",
        instructions=FOCUSED_EXTRACTION_SYSTEM_PROMPT,
        input_messages=[{"role": "user", "content": f"지목된 인물: {character_name}\n\n{combined}"}],
        max_output_tokens=8000,
        response_model=CharacterProfileData,
        input_policy="training",
    )
