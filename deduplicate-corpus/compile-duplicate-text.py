from modules.compile_duplicate_text import compile_duplicate_text

paths = {
    "deduped_dir": "objectives/deduplicate-corpus/deduped",
    "path_to_output": "objectives/deduplicate-corpus/intermediate/duplicate-text.text"
}
deduped_dir, path_to_output = paths.values()

compile_duplicate_text(deduped_dir, path_to_output)