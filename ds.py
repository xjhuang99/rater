import pandas as pd
import openai
import time
import json
import re
import os
import concurrent.futures
from tqdm import tqdm  # For progress bar display

# Configuration Parameters
CONFIG = {
    "DEEPSEEK_API_KEY": "xxx",
    "INPUT_EXCEL_PATH": "Study3.xlsx",
    "OUTPUT_EXCEL_PATH": "Study3_ds3.xlsx",
    "SHEET_NAME": "Sheet1",
    "CHAT_CONTENT_COLUMN": "merged_messages",
    "PROLIFIC_ID_COLUMN": "PROLIFIC_PID",  
    "MAX_RETRIES": 3,
    "API_TIMEOUT": 60,
    "DEEPSEEK_MODEL": "deepseek-chat",
    "PARALLEL_WORKERS": 3,  # Number of parallel threads
    "OUTPUT_BATCH_SIZE": 100  # Batch processing size
}


def initialize_deepseek_client():
    """Initialize DeepSeek API client"""
    return openai.OpenAI(
        api_key=CONFIG["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1"  # Official DeepSeek API endpoint
    )


def rate_chat_with_deepseek(client, chat_content: str, prolific_id: str):
    """Call DeepSeek model to get ratings, return None on failure"""
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            response = client.chat.completions.create(
                model=CONFIG["DEEPSEEK_MODEL"],
                messages=[
                    {"role": "system", "content": """
You are a feedback analysis expert. Your task is to analyze essay feedback and rate the feedback provider on three dimensions using a 1-7 scale (1 = Very Low, 4 = Moderate, 7 = Very High).

1. Competence - How competent, confident, intelligent, capable, and skillful do they seem based on their comments?
2. Warmth - How warm, friendly, good-natured, and tolerant do they appear in their tone and approach?
3. Trustworthiness - How sincere, trustworthy, and honest do they seem in their evaluation?
4. PROLIFIC_PID: The participant ID of this chat.

Return STRICTLY a JSON object with these EXACT field names.
Do NOT include any other text, comments, or markdown formatting (no ```json or ```).

{
    "competence": 1-7,
    "warmth": 1-7,
    "trustworthiness": 1-7,
    "reason": "Brief explanation for the ratings.",
    "prolific_pid": "The participant ID of this chat"
}
"""},
                    {"role": "user", "content": f"Rate this feedback (PROLIFIC_PID: {prolific_id}):\n{chat_content}"}
                ],
                temperature=0.1,
                max_tokens=300,
                timeout=CONFIG["API_TIMEOUT"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error (Attempt {attempt + 1}, PROLIFIC_PID:{prolific_id}): {str(e)}")
            if attempt < CONFIG["MAX_RETRIES"] - 1:
                time.sleep(10 if "rate limit" in str(e).lower() else 5)
    return None


def clean_response_text(text):
    """Clean response text by removing Markdown and non-JSON content"""
    if not text:
        return text

    text = re.sub(r'^```(json)?\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    text = re.sub(r'^\{"name":"JSONPlugin","parameters":\{"input":"JSONStringify\[', '', text)
    text = re.sub(r'\]\}"\}\]$', '', text)
    text = text.replace('\\"', '"')
    return text.strip()


def parse_rating_response(response_text: str):
    """Parse rating response, return None on failure"""
    if not response_text:
        return None, None

    cleaned_text = clean_response_text(response_text)
    try:
        if not (cleaned_text.startswith("{") and cleaned_text.endswith("}")):
            raise ValueError(f"Invalid JSON format: {cleaned_text[:50]}...")

        rating = json.loads(cleaned_text)
        required_fields = ["competence", "warmth", "trustworthiness", "reason", "prolific_pid"]
        for field in required_fields:
            if field not in rating:
                raise ValueError(f"Missing field: {field}")

        # Validate rating ranges
        for dimension in ["competence", "warmth", "trustworthiness"]:
            value = rating[dimension]
            if not (1 <= value <= 7):
                raise ValueError(f"{dimension} rating must be between 1-7")

        return {
            "competence": rating["competence"],
            "warmth": rating["warmth"],
            "trustworthiness": rating["trustworthiness"],
            "reason": rating["reason"],
            "prolific_pid": rating["prolific_pid"]
        }, None

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}, response: {cleaned_text[:50]}...")
        return None, f"JSON decoding error: {str(e)}"
    except Exception as e:
        print(f"Parsing failed: {str(e)}, response: {cleaned_text[:50]}...")
        return None, f"Parsing failed: {str(e)}"


def process_row(client, row, save_dir):
    """Process a single row of data and return results"""
    prolific_id = str(row[CONFIG["PROLIFIC_ID_COLUMN"]]).strip()
    content = str(row[CONFIG["CHAT_CONTENT_COLUMN"]]).strip()

    if not content:
        print(f"Skipping PROLIFIC_PID:{prolific_id}: Empty content")
        return row.name, {
            "competence": None, "warmth": None, "trustworthiness": None,
            "reason": "Empty content", "prolific_pid": prolific_id
        }

    response = rate_chat_with_deepseek(client, content, prolific_id)

    # Save raw API response
    api_file = os.path.join(save_dir, f"api_response_{prolific_id}.txt")
    with open(api_file, "w", encoding="utf-8") as f:
        f.write(response if response else "No response")

    ratings, error = parse_rating_response(response)
    if ratings:
        if ratings["prolific_pid"] != prolific_id:
            print(f"Warning PROLIFIC_PID:{prolific_id}: Returned ID mismatch ({ratings['prolific_pid']})")
        print(f"Success PROLIFIC_PID:{prolific_id}: {ratings}")
        return row.name, ratings
    else:
        print(f"Failed PROLIFIC_PID:{prolific_id}: {error or 'Unknown error'}")
        return row.name, {
            "competence": None, "warmth": None, "trustworthiness": None,
            "reason": error or "Parsing failed", "prolific_pid": prolific_id
        }


def process_chat_ratings():
    """Main function to process all chat feedback ratings"""
    client = initialize_deepseek_client()
    save_dir = "/Users/huangxinjie/Desktop/personality/save2"
    os.makedirs(save_dir, exist_ok=True)

    try:
        df = pd.read_excel(CONFIG["INPUT_EXCEL_PATH"], sheet_name=CONFIG["SHEET_NAME"])
    except Exception as e:
        print(f"Excel read error: {str(e)}")
        return

    # Verify required columns exist
    required_cols = [CONFIG["PROLIFIC_ID_COLUMN"], CONFIG["CHAT_CONTENT_COLUMN"]]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'")
            return

    # Create result columns
    df[["competence", "warmth", "trustworthiness", "reason", "prolific_pid"]] = None
    total_rows = len(df)
    print(f"Starting to process {total_rows} records with {CONFIG['PARALLEL_WORKERS']} parallel threads")

    progress_bar = tqdm(total=total_rows, desc="Processing progress")
    processed = 0

    # Process data in batches
    for batch_start in range(0, total_rows, CONFIG["OUTPUT_BATCH_SIZE"]):
        batch_end = min(batch_start + CONFIG["OUTPUT_BATCH_SIZE"], total_rows)
        batch_df = df.iloc[batch_start:batch_end]

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["PARALLEL_WORKERS"]) as executor:
            future_to_index = {
                executor.submit(process_row, client, row, save_dir): row.name
                for _, row in batch_df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_index):
                index, result = future.result()
                df.loc[index, ["competence", "warmth", "trustworthiness", "reason", "prolific_pid"]] = [
                    result["competence"], result["warmth"], result["trustworthiness"],
                    result["reason"], result["prolific_pid"]
                ]
                processed += 1
                progress_bar.update(1)

        # Save intermediate results after each batch
        print(f"Processed {processed}/{total_rows} records, saving intermediate results...")
        save_columns = [
            CONFIG["PROLIFIC_ID_COLUMN"], CONFIG["CHAT_CONTENT_COLUMN"],
            "competence", "warmth", "trustworthiness", "reason"
        ]
        df[save_columns].to_excel(CONFIG["OUTPUT_EXCEL_PATH"], index=False)

    progress_bar.close()
    print(f"\n✅ Processing completed. Results saved to: {CONFIG['OUTPUT_EXCEL_PATH']}")


if __name__ == "__main__":
    process_chat_ratings()
