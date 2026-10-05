"""Shared validation for character and world training sources."""

from fastapi import HTTPException, Request, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.formparsers import MultiPartException, MultiPartParser


MAX_SOURCE_CHARS = 300_000
# UTF-8 uses at most four bytes per Unicode character. Bound reads before decoding.
MAX_SOURCE_BYTES = MAX_SOURCE_CHARS * 4 + 3  # Optional UTF-8 BOM.
MEMORY_SPOOL_BYTES = 5_000_000  # Above Vercel's 4.5 MB function payload ceiling.


def _bad_source(code: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"code": code})


async def read_training_source(text: str | None, file: UploadFile | None) -> str:
    """A selected file takes precedence over text, matching the existing UI policy."""
    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".txt"):
            raise _bad_source("invalid_source_file_type")
        content = await file.read(MAX_SOURCE_BYTES + 1)
        if not content:
            raise _bad_source("empty_training_file")
        if len(content) > MAX_SOURCE_BYTES:
            raise _bad_source("source_too_long")
        try:
            source = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise _bad_source("source_invalid_utf8") from exc
        if not source.strip():
            raise _bad_source("empty_training_file")
    else:
        source = text or ""
        if not source.strip():
            raise _bad_source("training_source_missing")

    if len(source) > MAX_SOURCE_CHARS:
        raise _bad_source("source_too_long")
    return source


async def parse_training_request(request: Request) -> dict:
    """Accept JSON text or multipart files without Starlette's 1 MiB text-field default."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type == "application/json":
        try:
            fields = await request.json()
        except ValueError as exc:
            raise _bad_source("malformed_training_request") from exc
        if not isinstance(fields, dict):
            raise _bad_source("malformed_training_request")
        file = None
    elif content_type in ("multipart/form-data", "application/x-www-form-urlencoded"):
        try:
            if content_type == "multipart/form-data":
                parser = MultiPartParser(
                    request.headers, request.stream(), max_files=1, max_fields=4,
                    max_part_size=MAX_SOURCE_BYTES + 4096,
                )
                # Valid training uploads stay in RAM rather than UploadFile's 1 MiB disk spool.
                parser.spool_max_size = MEMORY_SPOOL_BYTES
                form = await parser.parse()
            else:
                form = await request.form(max_fields=4, max_part_size=MAX_SOURCE_BYTES + 4096)
        except (MultiPartException, StarletteHTTPException, ValueError) as exc:
            raise _bad_source("malformed_training_request") from exc
        try:
            fields = dict(form)
            file = fields.get("file")
            if file is not None and not isinstance(file, StarletteUploadFile):
                raise _bad_source("malformed_training_request")
            if any(not isinstance(value, str) for key, value in fields.items() if key != "file"):
                raise _bad_source("malformed_training_request")
            fields["raw_text"] = await read_training_source(fields.get("text"), file)
            fields["_source_type"] = "file" if file is not None else "text"
            return fields
        finally:
            await form.close()
    else:
        raise _bad_source("malformed_training_request")

    if not isinstance(fields.get("text"), (str, type(None))):
        raise _bad_source("malformed_training_request")
    fields["raw_text"] = await read_training_source(fields.get("text"), file)
    fields["_source_type"] = "text"
    return fields
