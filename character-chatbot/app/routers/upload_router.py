import os
import time
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from ..auth import get_current_user_id

router = APIRouter()
UPLOAD_DIR = "uploads"


@router.post("/")
def upload_image(image: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = image.filename.replace(" ", "_") if image.filename else "upload"
    filename = f"{int(time.time() * 1000)}-{safe_name}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(image.file, f)

    return {"url": f"/uploads/{filename}"}