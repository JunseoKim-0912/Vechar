export const MAX_SOURCE_CHARS = 300_000;
const MAX_SOURCE_BYTES = MAX_SOURCE_CHARS * 4 + 3; // Optional UTF-8 BOM.

function sourceError(code) {
  const error = new Error(code);
  error.code = code;
  return error;
}

export async function trainingRequestBody({ sourceType, text, file, seriesName, episodeNumber }) {
  const fields = { source_type: sourceType };
  if (sourceType === "NOVEL_EPISODE") {
    fields.series_name = seriesName;
    fields.episode_number = episodeNumber;
  }

  // A selected file takes precedence over any text left in the textarea.
  if (file) {
    if (!file.name?.toLowerCase().endsWith(".txt")) throw sourceError("invalid_source_file_type");
    if (!file.size) throw sourceError("empty_training_file");
    if (file.size > MAX_SOURCE_BYTES) throw sourceError("source_too_long");
    let source;
    try {
      source = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
    } catch {
      throw sourceError("source_invalid_utf8");
    }
    if (!source.trim()) throw sourceError("empty_training_file");
    if (Array.from(source).length > MAX_SOURCE_CHARS) throw sourceError("source_too_long");
    const form = new FormData();
    for (const [key, value] of Object.entries(fields)) form.append(key, value);
    form.append("file", file);
    return form;
  } else {
    if (!text?.trim()) throw sourceError("training_source_missing");
    if (Array.from(text).length > MAX_SOURCE_CHARS) throw sourceError("source_too_long");
    return { ...fields, text };
  }
}
