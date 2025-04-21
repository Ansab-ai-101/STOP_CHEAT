import os
import json
import torch
import clip
import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError # Explicitly import UnidentifiedImageError
import gradio as gr
import openai
import requests
import sqlite3
from tqdm import tqdm
from io import BytesIO
from datetime import datetime
from pathlib import Path
import base64

# FastAPI and Pydantic imports
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
# import uvicorn # Not needed if run by external server like HF Spaces

# ─────────────────────────────────────────────
# 🧠 STEP 1: LOAD CLIP MODEL
# ─────────────────────────────────────────────
# Moved model loading into a function to control execution
_model = None
_preprocess = None
_device = None

def load_clip_model():
    """Loads the CLIP model and preprocess function."""
    global _model, _preprocess, _device
    if _model is None:
        try:
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {_device}")
            _model, _preprocess = clip.load("ViT-B/32", device=_device)
            print("✅ CLIP Model loaded successfully.")
        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to load CLIP model: {e}")
            # Depending on severity, you might want to raise an error or exit
            # For now, let it continue but model-dependent parts will fail
            _model = None
            _preprocess = None
            _device = None
    return _model, _preprocess, _device

# ─────────────────────────────────────────────
# 📁 STEP 2: PATH CONFIGURATION
# ─────────────────────────────────────────────
# Define the mount path used for the persistent disk on Render
# --- IMPORTANT: Ensure this matches the 'Mount Path' set in Render service settings ---
RENDER_DISK_MOUNT_PATH = "/data"
DEFAULT_DB_PATH = os.path.join(RENDER_DISK_MOUNT_PATH, "tinder_profiles.db")

# Define the path for the default profiles JSON file
# Assumes 'profiles.json' is in the root of your Git repository alongside app.py
# '.' represents the current working directory (repo root on Render)
APP_ROOT_DIR = "."
DEFAULT_JSON_PATH = os.path.join(APP_ROOT_DIR, "profiles.json")

# Ensure the directory for the database exists on the persistent disk
# This helps prevent errors if the directory isn't automatically created on first boot
try:
    os.makedirs(RENDER_DISK_MOUNT_PATH, exist_ok=True)
    print(f"✅ Ensured directory exists: {RENDER_DISK_MOUNT_PATH}")
except OSError as e:
    # Log a warning but continue, database connection might still succeed or fail later
    print(f"⚠️ Warning: Could not create directory {RENDER_DISK_MOUNT_PATH}: {e}. Check disk mount and permissions.")

print(f"Database path set to: {DEFAULT_DB_PATH}")
print(f"Default JSON path set to: {DEFAULT_JSON_PATH} (Ensure this file exists in your repo if needed)")

# ─────────────────────────────────────────────
# 🗄️ STEP 3: DATABASE SETUP
# ─────────────────────────────────────────────
def setup_database(db_path=DEFAULT_DB_PATH):
    """Initialize SQLite database with required tables"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            bio TEXT,
            added_date TEXT
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT,
            url TEXT UNIQUE,
            embedding BLOB,
            FOREIGN KEY (profile_id) REFERENCES profiles(id)
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT,
            import_date TEXT,
            profiles_count INTEGER,
            photos_count INTEGER
        )
        ''')
        conn.commit()
        conn.close()
        print(f"✅ Database initialized/verified at {db_path}")
        return db_path
    except Exception as e:
        print(f"❌ ERROR: Failed to setup database at {db_path}: {e}")
        return None

# ─────────────────────────────────────────────
# 📦 STEP 4: PROFILE DATA MANAGEMENT
# ─────────────────────────────────────────────
def load_profile_data(json_file_path=None, json_data=None):
    """Load profile data either from a file or directly from JSON data"""
    profiles = None
    source_type = "unknown"
    try:
        if json_data:
            profiles = json_data
            source_type = "json_data input"
        elif json_file_path and os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
            source_type = f"file: {json_file_path}"
        elif os.path.exists(DEFAULT_JSON_PATH):
             print(f"Attempting to load from default path: {DEFAULT_JSON_PATH}")
             with open(DEFAULT_JSON_PATH, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
             source_type = f"default file: {DEFAULT_JSON_PATH}"
        else:
             print("⚠️ No profile data source found (json_data, json_file_path, or default path).")
             profiles = []

        if profiles is not None:
            print(f"✅ Loaded {len(profiles)} profiles from {source_type}")
        return profiles if profiles is not None else []

    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format in {source_type}: {e}")
        return []
    except Exception as e:
        print(f"❌ ERROR: Failed to load profile data from {source_type}: {e}")
        return []


def store_profiles_in_db(profiles, db_path=DEFAULT_DB_PATH, source_name="New Import"):
    """Store profiles in the SQLite database"""
    if not db_path:
         print("❌ ERROR: Database path not valid for storing profiles.")
         return 0, 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    new_profiles = 0
    new_photos = 0
    updated_profiles = 0
    skipped_profiles = 0

    print(f"Storing {len(profiles)} profiles from source: {source_name}")
    for profile in tqdm(profiles, desc="Storing profiles"):
        try:
            profile_id = profile.get("Id")
            if not profile_id:
                profile_id = str(hash(profile.get("Name", "") + str(profile.get("Age", 0)) + profile.get("Bio", "")))
                print(f"⚠️ Warning: Profile missing 'Id'. Generated ID: {profile_id} for Name: {profile.get('Name')}")

            name = profile.get("Name", "Unknown")
            age = profile.get("Age")
            bio = profile.get("Bio", "")

            cursor.execute("SELECT id FROM profiles WHERE id=?", (profile_id,))
            exists = cursor.fetchone()

            if exists:
                cursor.execute(
                    "UPDATE profiles SET name=?, age=?, bio=?, added_date=? WHERE id=?",
                    (name, age, bio, today, profile_id)
                )
                updated_profiles += 1
            else:
                cursor.execute(
                    "INSERT INTO profiles (id, name, age, bio, added_date) VALUES (?, ?, ?, ?, ?)",
                    (profile_id, name, age, bio, today)
                )
                new_profiles += 1

            for photo_url in profile.get("Photos", []):
                if not photo_url or not isinstance(photo_url, str):
                    print(f"⚠️ Skipping invalid photo URL for profile {profile_id}: {photo_url}")
                    continue
                cursor.execute("SELECT photo_id FROM photos WHERE url=?", (photo_url,))
                photo_exists = cursor.fetchone()

                if not photo_exists:
                    cursor.execute(
                        "INSERT INTO photos (profile_id, url, embedding) VALUES (?, ?, NULL)",
                        (profile_id, photo_url)
                    )
                    new_photos += 1

        except Exception as e:
            print(f"❌ ERROR storing profile (ID: {profile.get('Id', 'N/A')}, Name: {profile.get('Name', 'N/A')}): {e}")
            skipped_profiles += 1

    if new_profiles > 0 or new_photos > 0:
        try:
            cursor.execute(
                "INSERT INTO data_sources (source_name, import_date, profiles_count, photos_count) VALUES (?, ?, ?, ?)",
                (source_name, today, new_profiles, new_photos)
            )
        except Exception as e:
            print(f"❌ ERROR logging import to data_sources table: {e}")

    conn.commit()
    conn.close()
    print(f"✅ Profile storing complete. New: {new_profiles}, Updated: {updated_profiles}, Skipped: {skipped_profiles}. New Photos: {new_photos}")
    return new_profiles, new_photos

# ─────────────────────────────────────────────
# 🖼️ STEP 5: IMAGE PROCESSING & EMBEDDINGS
# ─────────────────────────────────────────────
def download_and_process_image(url):
    """Download image from URL and return PIL Image"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, headers=headers, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type')
        if content_type and not content_type.startswith('image/'):
             print(f"⚠️ Warning: URL {url} returned non-image content-type: {content_type}")
             # Allow processing anyway, PIL might handle it or raise UnidentifiedImageError

        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error downloading image from {url}: {e}")
        return None
    except UnidentifiedImageError:
         print(f"⚠️ PIL UnidentifiedImageError: Cannot identify image file from {url}. Maybe not an image?")
         return None
    except Exception as e:
        print(f"⚠️ Error processing image from {url}: {e}")
        return None


def generate_and_store_embeddings(db_path=DEFAULT_DB_PATH, max_images=-1):
    """Generate CLIP embeddings for profile images and store in database"""
    model, preprocess, device = load_clip_model() # Ensure model is loaded
    if not model or not preprocess:
        print("❌ Cannot generate embeddings: CLIP model not loaded.")
        return 0, 0

    if not db_path:
        print("❌ ERROR: Database path not valid for generating embeddings.")
        return 0, 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT p.photo_id, p.url, pr.id, pr.name, pr.age, pr.bio
        FROM photos p
        JOIN profiles pr ON p.profile_id = pr.id
        WHERE p.embedding IS NULL
    """
    params = ()
    if max_images > 0:
        query += " LIMIT ?"
        params = (max_images,)

    cursor.execute(query, params)
    photos_to_process = cursor.fetchall()

    if not photos_to_process:
        print("✅ No new images require embedding.")
        conn.close()
        return 0, 0

    processed_count = 0
    error_count = 0
    batch_size = 16
    photo_batch = []

    print(f"🧠 Generating CLIP embeddings for {len(photos_to_process)} new images...")
    for photo_data in tqdm(photos_to_process, desc="Processing images"):
        photo_id, url, profile_id, name, age, bio = photo_data
        try:
            img = download_and_process_image(url)
            if img is None:
                error_count += 1
                continue

            img_input = preprocess(img).unsqueeze(0)
            photo_batch.append({'id': photo_id, 'input': img_input, 'url': url})

            if len(photo_batch) >= batch_size or photo_data == photos_to_process[-1]:
                if not photo_batch: continue

                batch_inputs = torch.cat([p['input'] for p in photo_batch]).to(device)
                with torch.no_grad():
                    batch_embeddings = model.encode_image(batch_inputs).cpu().numpy()

                batch_embeddings /= np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

                for i, photo_info in enumerate(photo_batch):
                    emb = batch_embeddings[i].flatten()
                    cursor.execute(
                        "UPDATE photos SET embedding = ? WHERE photo_id = ?",
                        (emb.tobytes(), photo_info['id'])
                    )
                    processed_count += 1

                conn.commit()
                photo_batch = []

        except Exception as e:
            print(f"❌ ERROR processing image (ID: {photo_id}, URL: {url}): {e}")
            error_count += 1
            photo_batch = []

    conn.close()
    print(f"✅ Finished embedding generation. Processed: {processed_count}, Errors: {error_count}.")
    return processed_count, error_count


def load_embeddings_from_db(db_path=DEFAULT_DB_PATH):
    """Load all embeddings, urls and profile info from the database"""
    if not db_path or not os.path.exists(db_path):
        print(f"⚠️ Database file not found at {db_path}. Cannot load embeddings.")
        return np.array([]).astype("float32"), [], []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    embeddings = []
    image_urls = []
    profile_info = []

    try:
        cursor.execute("""
            SELECT p.embedding, p.url, pr.id, pr.name, pr.age, pr.bio
            FROM photos p
            JOIN profiles pr ON p.profile_id = pr.id
            WHERE p.embedding IS NOT NULL
        """)
        result = cursor.fetchall()
    except Exception as e:
        print(f"❌ ERROR querying embeddings from database: {e}")
        result = []
    finally:
        conn.close()

    expected_dimension = 512 # For ViT-B/32
    for row in result:
        embedding_bytes, url, profile_id, name, age, bio = row
        if embedding_bytes:
            try:
                emb = np.frombuffer(embedding_bytes, dtype=np.float32)
                if emb.shape[0] == expected_dimension:
                    embeddings.append(emb)
                    image_urls.append(url)
                    profile_info.append({
                        "Id": profile_id, "Name": name, "Age": age,
                        "Bio": bio, "PhotoUrl": url
                    })
                else:
                    print(f"⚠️ Skipping embedding for URL {url} due to unexpected dimension: {emb.shape[0]} (expected {expected_dimension})")
            except Exception as e:
                 print(f"⚠️ Error processing embedding blob for URL {url}: {e}")

    if embeddings:
        embeddings_array = np.vstack(embeddings).astype("float32")
    else:
        embeddings_array = np.array([]).astype("float32")

    print(f"📊 Loaded {len(embeddings_array)} valid embeddings from database")
    return embeddings_array, image_urls, profile_info


def get_database_stats(db_path=DEFAULT_DB_PATH):
    """Get statistics about the database"""
    stats = {
        "total_profiles": 0, "total_photos": 0, "processed_photos": 0,
        "recent_imports": []
    }
    if not db_path or not os.path.exists(db_path):
         print(f"⚠️ Database file not found at {db_path}. Cannot get stats.")
         return stats

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM profiles")
        stats["total_profiles"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM photos")
        stats["total_photos"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding IS NOT NULL")
        stats["processed_photos"] = cursor.fetchone()[0]
        cursor.execute("""
            SELECT source_name, import_date, profiles_count, photos_count
            FROM data_sources ORDER BY source_id DESC LIMIT 5
        """)
        stats["recent_imports"] = cursor.fetchall()
    except Exception as e:
        print(f"❌ ERROR getting database stats: {e}")
    finally:
        conn.close()
    return stats

# ─────────────────────────────────────────────
# ⚡ STEP 6: BUILD FAISS INDEX
# ─────────────────────────────────────────────
def build_faiss_index(embeddings):
    """Build FAISS index from embeddings"""
    if embeddings is None or embeddings.shape[0] == 0:
        print("⚠️ Cannot build FAISS index: No embeddings provided.")
        return None
    try:
        dimension = embeddings.shape[1]
        expected_dimension = 512 # For ViT-B/32
        if dimension != expected_dimension:
             print(f"❌ ERROR: Embeddings have incorrect dimension {dimension} (expected {expected_dimension}). Cannot build index.")
             return None
        index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity
        index.add(embeddings)
        print(f"✅ FAISS index built successfully with {index.ntotal} vectors.")
        return index
    except Exception as e:
        print(f"❌ ERROR building FAISS index: {e}")
        return None

# ─────────────────────────────────────────────
# 🔐 STEP 7: OPENAI API SETUP
# ─────────────────────────────────────────────
_openai_client = None

def init_openai():
    """Initialize OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY environment variable not found. GPT analysis will be disabled.")
            return None
        try:
            _openai_client = openai.OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized.")
        except Exception as e:
            print(f"❌ ERROR initializing OpenAI client: {e}")
            _openai_client = None
    return _openai_client

# ─────────────────────────────────────────────
# 🔎 STEP 8: SEARCH FUNCTIONALITY
# ─────────────────────────────────────────────
def search_similar_faces(user_image, index, all_embeddings, image_urls, profile_info, top_k=50, min_score=0.80):
    """Search for similar faces using CLIP + FAISS with minimum score threshold"""
    model, preprocess, device = load_clip_model() # Ensure model is available
    if not model or not preprocess:
        print("❌ Search failed: CLIP model not loaded.")
        return [], [], "Error/100"

    if index is None or not hasattr(index, 'search') or index.ntotal == 0:
        print("⚠️ Search failed: FAISS index is not available or empty.")
        return [], [], "0/100"

    if user_image is None:
         print("⚠️ Search failed: No user image provided.")
         return [], [], "0/100"

    try:
        user_image = user_image.convert("RGB")
        tensor = preprocess(user_image).unsqueeze(0).to(device)
        with torch.no_grad():
            query_emb = model.encode_image(tensor).cpu().numpy().astype("float32")
            query_emb /= np.linalg.norm(query_emb) # Normalize query embedding
    except Exception as e:
        print(f"❌ ERROR: Image preprocessing or embedding failed: {e}")
        return [], [], "Error/100"

    try:
        scores, indices = index.search(query_emb, min(top_k, index.ntotal))
        scores = scores.flatten()
        indices = indices.flatten()

        match_details_list = []
        valid_indices = indices[scores >= min_score]
        valid_scores = scores[scores >= min_score]

        print(f"Found {len(valid_indices)} matches with score >= {min_score} (out of {top_k} checked).")

        for i in range(len(valid_indices)):
            idx = valid_indices[i]
            score = valid_scores[i]
            if idx < 0 or idx >= len(image_urls) or idx >= len(profile_info):
                 print(f"⚠️ Skipping invalid index {idx} from FAISS results.")
                 continue
            try:
                url = image_urls[idx]
                info = profile_info[idx]
                match_details_list.append({"url": url, "score": float(score), "info": info})
            except Exception as e:
                print(f"⚠️ Error processing match details for index {idx} (URL: {url}): {e}")

        # Calculate risk score based on filtered matches
        risk_score_display = "0/100"
        if match_details_list:
             risk_value = np.mean([d["score"] for d in match_details_list])
             risk_score_display = f"{min(100, int(risk_value * 100))}/100"

        # Download images ONLY if needed (e.g., for Gradio Gallery)
        # For API, we don't need PIL images, only details
        matching_images_pil = []
        # Uncomment if PIL images are needed by the caller (like Gradio UI)
        # for detail in match_details_list:
        #      img = download_and_process_image(detail["url"])
        #      if img:
        #           matching_images_pil.append(img)

        return matching_images_pil, match_details_list, risk_score_display

    except Exception as e:
        print(f"❌ ERROR during FAISS search or result processing: {e}")
        return [], [], "Error/100"


# ─────────────────────────────────────────────
# 🧠 STEP 9: GPT-4 ANALYSIS
# ─────────────────────────────────────────────
def generate_gpt4_analysis(match_details):
    """Generate profile analysis using GPT-4"""
    openai_client = init_openai() # Ensure client is initialized
    if not openai_client:
        return "GPT analysis disabled (OpenAI client not initialized or API key missing)"
    if not match_details:
        return "No high-similarity matches found (score >= 0.80) to analyze."

    max_matches_for_gpt = 5
    limited_matches = match_details[:max_matches_for_gpt]
    try:
        profiles_info = []
        for d in limited_matches:
            info = d['info']
            profiles_info.append({
                "name": info.get('Name', 'N/A'), "age": info.get('Age', 'N/A'),
                "bio": info.get('Bio', 'N/A')[:150], "similarity": f"{d['score']:.3f}"
            })

        if not profiles_info:
             return "No valid profile information found in matches for analysis."

        prompt = ( f"Analyze the authenticity of a dating profile based on the following {len(profiles_info)} "
                   f"high-similarity face matches (score >= 0.80) found in a database for a submitted photo. "
                   f"Matches:\n\n" )
        for i, profile in enumerate(profiles_info, 1):
            prompt += ( f"Match {i}:\n- Name: {profile['name']}\n- Age: {profile['age']}\n"
                        f"- Similarity: {profile['similarity']}\n- Bio Snippet: {profile['bio']}\n\n" )
        prompt += ( "Based ONLY on these matches, provide a concise analysis addressing:\n"
                    "1. Likelihood of the submitted photo being associated with fake/catfish profiles?\n"
                    "2. Likelihood of the person simply having multiple profiles?\n"
                    "3. Any red flags or safety advice.\n"
                    "Focus on evidence from the provided matches. Be professional and objective." )

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[ {"role": "system", "content": "You are a dating profile authenticity analyst. Analyze potential photo reuse based *only* on the provided face match data. Be concise and objective."},
                       {"role": "user", "content": prompt} ],
            max_tokens=250, temperature=0.5
        )
        analysis = response.choices[0].message.content
        return analysis.strip()

    except Exception as e:
        print(f"❌ ERROR during GPT analysis: {e}")
        return f"(OpenAI analysis failed: {e})"

# ─────────────────────────────────────────────
# 🎛️ STEP 10: APPLICATION CLASS (TinderScanner)
# ─────────────────────────────────────────────
class TinderScanner:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        self.index = None
        self.all_embeddings = np.array([]).astype("float32")
        self.image_urls = []
        self.profile_info = []

        if not self.db_path:
             print("❌ Scanner initialized with invalid DB path. Database operations will fail.")
             return # Allow creation but flag as non-functional

        # Ensure database exists (call setup only once ideally, handled here)
        setup_database(self.db_path)
        # Load initial state
        self.init_from_database()

    def init_from_database(self):
        """Load embeddings from DB and build index"""
        print("Initializing scanner state from database...")
        self.all_embeddings, self.image_urls, self.profile_info = load_embeddings_from_db(self.db_path)
        if self.all_embeddings.shape[0] > 0:
            self.index = build_faiss_index(self.all_embeddings)
            if self.index:
                print(f"✅ Scanner ready with {self.index.ntotal} indexed photos.")
                return True
            else:
                 print("⚠️ Loaded embeddings but failed to build FAISS index.")
                 return False
        else:
            print("⚠️ Database contains no embeddings to load.")
            self.index = None
            return False

    def get_database_stats_text(self):
        """Get formatted database statistics"""
        stats = get_database_stats(self.db_path)
        processed_percent = (stats['processed_photos'] / stats['total_photos'] * 100) if stats['total_photos'] > 0 else 0
        text = (f"📊 DATABASE STATISTICS:\n"
                f"- Total Profiles: {stats['total_profiles']}\n"
                f"- Total Photos: {stats['total_photos']}\n"
                f"- Processed Photos (Embeddings): {stats['processed_photos']} ({processed_percent:.1f}%)\n\n"
                f"🔄 RECENT IMPORTS (Max 5):\n")
        if stats['recent_imports']:
            for source, date, profiles, photos in stats['recent_imports']:
                text += f"- {date} | {source}: {profiles} profiles, {photos} photos added\n"
        else:
            text += "- No import history found.\n"
        return text

    def load_data(self, json_text=None, source_name="JSON Import", json_file=None):
        """Load profile data from JSON, store, embed, and reload index."""
        print(f"--- Starting data load process from source: {source_name} ---")
        try:
            profiles_list = None
            if json_text:
                 try:
                      json_data = json.loads(json_text)
                      profiles_list = load_profile_data(json_data=json_data)
                 except json.JSONDecodeError as e:
                      return f"❌ Invalid JSON format provided: {e}"
            elif json_file:
                 profiles_list = load_profile_data(json_file_path=json_file)
            else: # Fallback to default
                 profiles_list = load_profile_data(json_file_path=DEFAULT_JSON_PATH)
                 if profiles_list: source_name = "Default profiles.json"

            if not profiles_list:
                 return "⚠️ No profile data loaded. Check input or default profiles.json."

            new_profiles, new_photos = store_profiles_in_db(profiles_list, self.db_path, source_name)
            processed_count, error_count = generate_and_store_embeddings(self.db_path)
            if error_count > 0: print(f"⚠️ Encountered {error_count} errors during embedding.")

            reloaded = self.init_from_database() # Reload state

            if reloaded:
                 stats = get_database_stats(self.db_path)
                 return (f"✅ Data processed successfully from '{source_name}'.\n"
                         f"- New Profiles: {new_profiles}, New Photos: {new_photos}, Embeddings Generated: {processed_count}\n"
                         f"- Database now contains {stats['total_profiles']} profiles and {stats['processed_photos']} processed photos.")
            else:
                 return ("⚠️ Data storing finished, but failed to initialize/reload scanner index. "
                         "Check logs for embedding errors.")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR during data load process: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed debug
            return f"❌ An unexpected error occurred during data loading: {e}"

    def scan_face(self, user_image, min_score=0.80):
        """UI Scan: Process image, find matches, return results for Gradio UI"""
        if user_image is None:
            return [], "", "0/100", "Please upload a face image."

        if not self.index:
            return [], "", "0/100", "Database index not ready. Load data first."

        print(f"Scanning face image. Index has {self.index.ntotal} vectors.")
        try:
            # Call core search (don't need PIL images returned from it directly)
            _, match_details_list, risk_score_display = search_similar_faces(
                user_image, self.index, self.all_embeddings, self.image_urls, self.profile_info,
                min_score=min_score
            )

            if not match_details_list:
                return [], "", "0/100", f"No matches found with similarity score >= {min_score}."

            # Download PIL images for the gallery based on results
            matching_images_pil = []
            for detail in match_details_list:
                 img = download_and_process_image(detail["url"])
                 if img:
                      matching_images_pil.append(img)

            # Format captions
            captions = [f"{d['info'].get('Name','N/A')} ({d['info'].get('Age','N/A')}) - Score: {d['score']:.3f}"
                        for d in match_details_list]
            match_details_text = "\n".join(captions)

            explanation = generate_gpt4_analysis(match_details_list)

            print(f"Scan complete. Found {len(matching_images_pil)} matches >= {min_score}.")
            return matching_images_pil, match_details_text, risk_score_display, explanation

        except Exception as e:
            print(f"❌ Error during scan_face execution: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed debug
            return [], "", "Error", f"An error occurred during scanning: {e}"

    def process_post_request(self, request_data):
        """API Request: Process POST request data, return JSON response"""
        print("--- Processing POST request ---")
        try:
            image_data = request_data.get("image_data")
            json_text = request_data.get("json_text")
            source_name = request_data.get("source_name", "API POST Import")
            uuid = request_data.get("uuid")
            min_score = request_data.get("min_score", 0.80)

            if json_text:
                print(f"Processing json_text provided in POST request (source: {source_name})...")
                load_result = self.load_data(json_text=json_text, source_name=source_name)
                if "❌ Error" in load_result or "⚠️" in load_result:
                     # Raise HTTP error if loading provided data fails
                     raise HTTPException(status_code=400, detail=f"Failed to load provided JSON data: {load_result}")

            user_image = None
            if image_data and isinstance(image_data, str):
                if image_data.startswith(('http://', 'https://')):
                    print(f"Downloading image from URL: {image_data[:100]}...")
                    user_image = download_and_process_image(image_data)
                else: # Assume base64
                    print("Decoding Base64 image data...")
                    try:
                        if "base64," in image_data: image_data = image_data.split("base64,")[1]
                        image_bytes = base64.b64decode(image_data)
                        user_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                        print("Base64 image decoded successfully.")
                    except Exception as e:
                         raise HTTPException(status_code=400, detail=f"Error decoding base64 image data: {e}")
            else:
                 raise HTTPException(status_code=400, detail="Missing or invalid 'image_data' (must be URL or Base64 string)")

            if user_image is None:
                 raise HTTPException(status_code=400, detail="Failed to load image from provided 'image_data'")

            if not self.index:
                 print("API Scan attempt failed: Index not available. Trying re-init...")
                 initialized = self.init_from_database()
                 if not initialized:
                      raise HTTPException(status_code=503, detail="Database index not ready. Load data first.")

            print(f"Scanning image via API. Index has {self.index.ntotal} vectors. Min score: {min_score}")
            _, match_details_list, risk_score_display = search_similar_faces(
                user_image, self.index, self.all_embeddings, self.image_urls, self.profile_info,
                min_score=min_score
            ) # Don't need PIL images for API

            explanation = generate_gpt4_analysis(match_details_list)
            captions = [f"{d['info'].get('Name','N/A')} ({d['info'].get('Age','N/A')}) - Score: {d['score']:.3f}"
                        for d in match_details_list]
            match_details_text_api = "\n".join(captions)

            response_payload = {
                "matches": match_details_list,
                "match_details_text": match_details_text_api,
                "risk_score": risk_score_display,
                "profile_analysis": explanation,
                "uuid": uuid
            }
            print(f"API Scan complete. Found {len(match_details_list)} matches. Returning response.")
            return response_payload

        except HTTPException as http_exc:
            print(f"HTTP Exception in POST request: {http_exc.status_code} - {http_exc.detail}")
            raise http_exc # Re-raise for FastAPI
        except Exception as e:
            print(f"❌ UNEXPECTED error processing POST request: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed debug
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# ─────────────────────────────────────────────
# ✨ STEP 11: FastAPI APP & ENDPOINT SETUP
# ─────────────────────────────────────────────

# Global variables for initialized components
scanner_instance = None
fastapi_app = None
gradio_ui = None

def initialize_scanner():
    """Initializes the TinderScanner instance."""
    global scanner_instance
    if scanner_instance is None:
        print("Instantiating TinderScanner...")
        try:
            # Ensure CLIP model is loaded before scanner uses it
            load_clip_model()
            # Ensure OpenAI client is initialized if needed later
            init_openai()
            scanner_instance = TinderScanner(db_path=DEFAULT_DB_PATH)
            # Scanner __init__ calls init_from_database() itself
            print("TinderScanner initialization process complete.")
        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to instantiate TinderScanner: {e}")
            scanner_instance = None
    return scanner_instance

def initialize_fastapi_app():
    """Creates and configures the FastAPI application."""
    global fastapi_app
    if fastapi_app is None:
        print("Creating FastAPI app...")
        fastapi_app = FastAPI(title="Tinder Scanner Pro API", version="1.0")
        add_fastapi_endpoints(fastapi_app)
        print("FastAPI app created.")
    return fastapi_app

def add_fastapi_endpoints(app_to_configure):
    """Adds API endpoints to the FastAPI app."""
    # Define Pydantic model for POST request body validation
    class PostMatchRequest(BaseModel):
        image_data: str = Field(..., description="Base64 encoded image string or image URL")
        json_text: str | None = Field(default=None, description="Optional JSON string of profiles to add/update")
        source_name: str = Field(default="API POST Import", description="Identifier for the data source if json_text is provided")
        uuid: str | None = Field(default=None, description="Optional unique identifier for the request")
        min_score: float | None = Field(default=0.80, ge=0.0, le=1.0, description="Minimum similarity score threshold (0.0 to 1.0)")

    @app_to_configure.post("/post_match", summary="Scan Image for Similar Faces")
    async def handle_post_match(payload: PostMatchRequest):
        """ API endpoint to scan an image and optionally add data. """
        if not scanner_instance: # Check the initialized global scanner
             raise HTTPException(status_code=503, detail="Scanner service not available due to initialization error.")
        request_dict = payload.model_dump()
        # process_post_request handles internal errors and raises HTTPException
        result = scanner_instance.process_post_request(request_dict)
        return result

    @app_to_configure.get("/", summary="Root / Health Check")
    async def read_root():
        """ Provides basic status information. """
        db_stats = get_database_stats(DEFAULT_DB_PATH)
        openai_client = init_openai() # Check current status
        return {
            "message": "Tinder Scanner API is running.",
            "database_status": {
                "profiles": db_stats.get("total_profiles", "N/A"),
                "total_photos": db_stats.get("total_photos", "N/A"),
                "processed_photos": db_stats.get("processed_photos", "N/A"),
            },
            "openai_status": "Initialized" if openai_client else "Disabled",
            "documentation": "/docs"
        }

# ─────────────────────────────────────────────
# 🖥️ STEP 12: GRADIO UI FUNCTION
# ─────────────────────────────────────────────

# Modify create_ui to accept scanner instance and use it
def create_gradio_ui_definition(scanner):
    """Defines the Gradio UI Blocks."""
    print("Defining Gradio UI...")
    if not scanner:
         print("ERROR: Cannot create Gradio UI because Scanner is not initialized.")
         with gr.Blocks(title="Error", theme=gr.themes.Soft()) as demo_error:
              gr.Markdown("# Application Initialization Error")
              gr.Markdown("The core scanner service failed to initialize. Please check the logs.")
         return demo_error

    # Use the passed 'scanner' instance for all UI interactions
    with gr.Blocks(title="Tinder Scanner Pro", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Tinder Scanner Pro – High-Similarity Face Matcher")
        gr.Markdown("Scan a face image to find high-similarity matches (default score >= 0.80) in Tinder profiles.")
        demo.queue()

        with gr.Tabs():
            with gr.TabItem("👤 Scan Face"):
                with gr.Row():
                    with gr.Column(scale=1):
                        user_image = gr.Image(type="pil", label="Upload Face Image", height=300)
                        scan_btn = gr.Button("Scan Face", variant="primary")
                        min_score_slider = gr.Slider(minimum=0.50, maximum=1.0, value=0.80, step=0.01, label="Minimum Similarity Score")
                        with gr.Accordion("Optional: Add Profiles During Scan", open=False):
                             new_json_ui = gr.Textbox( label="Append New JSON Data", placeholder='Paste JSON profiles here...', lines=3 )
                             source_name_ui = gr.Textbox( label="Source Name (if appending data)", value="UI Quick Add" )
                        uuid_input = gr.Textbox(label="UUID (Optional)", placeholder="Request identifier")
                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        matches_gallery = gr.Gallery(label="🔍 High-Similarity Matches", columns=4, height=400, object_fit="contain", preview=True)
                        with gr.Row():
                             risk_score = gr.Textbox(label="📊 Average Similarity Score") # Corrected label
                             uuid_output = gr.Textbox(label="UUID", interactive=False)
                        match_details = gr.Textbox(label="ℹ️ Match Details (Name, Age, Score)", lines=5, interactive=False)
                        gpt_analysis = gr.Textbox(label="🧠 AI Analysis (Authenticity Assessment)", lines=8, interactive=False)

                def ui_scan_wrapper(img, score_threshold, json_str, src_name, req_uuid):
                    load_msg = ""
                    if json_str:
                         load_msg = scanner.load_data(json_text=json_str, source_name=src_name) # Use passed scanner
                         print(f"UI Quick Add Result: {load_msg}")
                    # Call scanner's scan_face method
                    gallery_imgs, details_text, risk_val, analysis_text = scanner.scan_face(img, min_score=score_threshold)
                    return gallery_imgs, details_text, risk_val, analysis_text, req_uuid

                scan_btn.click( fn=ui_scan_wrapper,
                                inputs=[user_image, min_score_slider, new_json_ui, source_name_ui, uuid_input],
                                outputs=[matches_gallery, match_details, risk_score, gpt_analysis, uuid_output] )

            with gr.TabItem("🗃️ Database Management"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Import Profiles to Database")
                        json_input_db = gr.Textbox( label="Paste JSON Profile Data", placeholder='Format: [{"Id": "...", ...}]', lines=10 )
                        load_default_btn = gr.Button("Load from default profiles.json")
                        import_name_db = gr.Textbox( label="Import Source Name", value="Manual Import via UI" )
                        import_btn = gr.Button("Import Pasted Data to DB", variant="primary")
                        import_status = gr.Textbox(label="Import Status", lines=4, interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("### Database Statistics")
                        refresh_stats_btn = gr.Button("Refresh Statistics")
                        db_stats = gr.Textbox(label="Current Stats", lines=15, interactive=False)
                        db_stats.value = scanner.get_database_stats_text() # Initial value

                import_btn.click( fn=scanner.load_data, inputs=[json_input_db, import_name_db], outputs=[import_status]
                       ).then( fn=scanner.get_database_stats_text, outputs=[db_stats] ) # Refresh stats after import
                load_default_btn.click( fn=lambda: scanner.load_data(source_name="Default profiles.json"), inputs=[], outputs=[import_status]
                       ).then( fn=scanner.get_database_stats_text, outputs=[db_stats] ) # Refresh stats after import
                refresh_stats_btn.click( fn=scanner.get_database_stats_text, outputs=[db_stats] )

            with gr.TabItem("📖 API Documentation"):
                 api_doc_path = Path("api_docs.md")
                 api_doc_content = api_doc_path.read_text() if api_doc_path.exists() else "API Documentation file (api_docs.md) not found."
                 gr.Markdown(api_doc_content)

    print("Gradio UI definition complete.")
    return demo


# ─────────────────────────────────────────────
# 🚀 STEP 13: INITIALIZE & MOUNT APPLICATION
# ─────────────────────────────────────────────

# --- Main Execution Logic ---
# This block runs when the script is imported by the ASGI server (e.g., Uvicorn on HF Spaces)

# 1. Initialize Scanner (includes model loading, DB connection, index building)
scanner_instance = initialize_scanner() # Returns the instance or None

# 2. Initialize FastAPI App (creates app object and adds endpoints)
# The variable MUST be named "app" for Hugging Face Spaces discovery
app = initialize_fastapi_app() # Returns the FastAPI app instance

# 3. Define Gradio UI using the initialized scanner
gradio_ui = create_gradio_ui_definition(scanner_instance) # Returns Gradio Blocks or error UI

# 4. Mount Gradio onto FastAPI
if app and gradio_ui:
    print("Mounting Gradio UI onto FastAPI app at path '/'...")
    # Mount the UI definition onto the FastAPI app instance
    app = gr.mount_gradio_app(app, gradio_ui, path="/")
    print("Gradio UI mounted successfully.")
else:
    print("⚠️ Failed to mount Gradio UI due to initialization errors.")
    # FastAPI app ('app') might still be partially functional if it initialized

# The final 'app' object is now ready for the ASGI server.
# No need to call uvicorn.run() or demo.launch() when deploying on platforms like Hugging Face Spaces.
print("--- Application Setup Complete ---")
