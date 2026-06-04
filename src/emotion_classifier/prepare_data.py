# src/prepare_data.py
import os
import time
import pandas as pd
from groq import Groq
from config import EMOTION_SPLIT_PATH
import dotenv

dotenv.load_dotenv()

# ── 1. INITIALIZE GROQ CLIENT ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

LABEL_MAPPING = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def generate_synthetic_batch(emotion_name, batch_size):
    """
    Calls the Groq API with aggressive error handling and exponential backoff
    for rate limits (HTTP 429).
    """
    prompt = f"""
    You are an expert linguistic data annotator creating text data for an emotion classifier.
    Generate exactly {batch_size} unique, short, first-person conversational sentences or diary-like thoughts that implicitly or explicitly express the emotion of: '{emotion_name}'.
    
    CRITICAL STYLE GUIDELINES:
    1. Write entirely in lowercase. Do not include any punctuation (no periods, no commas).
    2. Keep sentences short (typically between 5 to 20 words).
    3. Use natural first-person phrasing (e.g., start with "i feel", "i am", "im feeling", "i just").
    4. Do not reuse the exact word '{emotion_name}' in every sentence; use situational context.
    5. Return ONLY the raw sentences, one per line. No numbers, no bullet points, no quotes, no markdown.
    """
    
    backoff_time = 30  # Start with a 30-second sleep if rate-limited
    
    while True:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=1500  # Extra breathing room for 50 sentences
            )
            
            raw_text = completion.choices[0].message.content.strip()
            lines = [line.strip().lower() for line in raw_text.split('\n') if line.strip()]
            clean_lines = [line.lstrip('0123456789.- ') for line in lines]
            
            return clean_lines[:batch_size]
            
        except Exception as e:
            error_msg = str(e).lower()
            # Catch standard 429 or rate limit strings
            if "429" in error_msg or "rate_limit" in error_msg or "rate limit" in error_msg:
                print(f"   ⚠️ Rate limit hit! Cooling down for {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 120)  # Exponential backoff, cap at 2 mins
            else:
                print(f"   ⚠️ Unexpected API Error: {e}")
                time.sleep(5)
                return []


def calculate_and_generate_balanced_data():
    """
    Calculates imbalance shortfall and uses an incremental checkpoint file 
    to seamlessly handle crashes, limits, and script restarts.
    """
    raw_train_path = EMOTION_SPLIT_PATH.format('train')
    balanced_out_path = EMOTION_SPLIT_PATH.format('train_balanced')
    checkpoint_path = EMOTION_SPLIT_PATH.format('synthetic_checkpoint')
    
    if not os.path.exists(raw_train_path):
        raise FileNotFoundError(f"❌ Could not find raw training data at {raw_train_path}")
        
    df = pd.read_csv(raw_train_path)
    class_counts = df['label'].value_counts()
    max_class_size = class_counts.max()
    
    # ── LOAD OR CREATE INCREMENTAL CHECKPOINT ────────────────────────────────
    if os.path.exists(checkpoint_path):
        print(f"🔄 Found existing checkpoint file: {checkpoint_path}")
        checkpoint_df = pd.read_csv(checkpoint_path)
        # Ensure values are loaded as a list of dicts
        synthetic_rows = checkpoint_df.to_dict(orient='records')
        print(f"📊 Loaded {len(synthetic_rows)} previously generated synthetic rows.")
    else:
        synthetic_rows = []
        
    print("=" * 60)
    print("🔮 RUNNING RESUME-CAPABLE SYNTHETIC BALANCING PIPELINE")
    print("=" * 60)
    print(f"Target Baseline Set by Majority Class: {max_class_size} rows\n")
    
    # Iterate over every class to balance the scale dynamically
    for label_idx, current_count in class_counts.items():
        emotion_name = LABEL_MAPPING[label_idx]
        
        # Calculate how many rows for this specific class are already in our checkpoint
        already_generated = sum(1 for row in synthetic_rows if row['label'] == label_idx)
        
        # True shortfall = Target - Original - What we already made in a past run
        shortfall = max_class_size - current_count - already_generated
        
        if shortfall <= 0:
            print(f"✅ Class '{emotion_name}' is fully balanced (Original: {current_count}, Synthetic: {already_generated}).")
            continue
            
        print(f"⚡ '{emotion_name}' needs {shortfall} more rows (Already generated: {already_generated}/{max_class_size - current_count}).")
        
        generated_for_class = 0
        step_chunk = 50  # Upgraded batch capacity to slash overall requests count
        
        while generated_for_class < shortfall:
            rem = shortfall - generated_for_class
            current_batch_size = min(step_chunk, rem)
            
            batch_texts = generate_synthetic_batch(emotion_name, current_batch_size)
            
            new_rows = []
            for text in batch_texts:
                if text and len(text.split()) > 2:  # Filter out empty or broken fragments
                    row = {"text": text, "label": label_idx}
                    synthetic_rows.append(row)
                    new_rows.append(row)
                    generated_for_class += 1
            
            # Flush immediately to checkpoint file so we never lose progress
            if new_rows:
                pd.DataFrame(synthetic_rows).to_csv(checkpoint_path, index=False)
            
            print(f"   -> Progress for [{emotion_name}]: {already_generated + generated_for_class}/{max_class_size - current_count}")
            time.sleep(2.0)  # Pacing delay between clean iterations
            
    # ── FINAL ASSEMBLY ───────────────────────────────────────────────────────
    if synthetic_rows:
        print("\n🧱 Combining original data with synthetic booster files...")
        synthetic_df = pd.DataFrame(synthetic_rows)
        balanced_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        # Shuffle completely so training loaders hit mixed labels
        balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        balanced_df.to_csv(balanced_out_path, index=False)
        
        # Clean up checkpoint file after clean completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
        print("=" * 60)
        print(f"🎉 Success! Final balanced dataset built cleanly.")
        print(f"📁 Export Path: {balanced_out_path}")
        print(f"📈 Dataset size updated from {len(df)} to {len(balanced_df)} rows.")
        print("=" * 60)
    else:
        print("\n✅ Dataset was already completely balanced!")


if __name__ == "__main__":
    calculate_and_generate_balanced_data()