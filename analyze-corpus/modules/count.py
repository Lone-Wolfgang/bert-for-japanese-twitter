import os
import json
from collections import Counter

def compile_token_counts(output_directory):
    combined_token_counts = Counter()

    # Get all the file paths in the output directory
    file_paths = [os.path.join(output_directory, file_name) for file_name in os.listdir(output_directory) if file_name.endswith(".json")]

    for file_path in file_paths:
        with open(file_path, "r") as f:
            token_counts = json.load(f)

        # Update the combined token counts with the token counts from the current file
        combined_token_counts.update(token_counts)

    # Sort the combined token counts by values in descending order
    sorted_token_counts = {k: v for k, v in sorted(
        combined_token_counts.items(), key=lambda item: item[1], reverse=True
    )}

    # Convert token counts to dictionary with decoded tokens
    decoded_token_counts = {
        token: count for token, count in sorted_token_counts.items()
    }

    # Save the combined token counts to a single JSON file
    combined_output_path = os.path.join(output_directory, "compiled_token_counts.json")
    with open(combined_output_path, "w") as f:
        json.dump(decoded_token_counts, f, indent=4)

    print(f"Compiled token counts saved to {combined_output_path}.")
    return
