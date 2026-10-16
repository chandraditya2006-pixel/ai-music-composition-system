"""
app.py
======
Flask Backend API for AI Music Composition System.
B.Tech Project: "Development of an AI-Based Music Composition System Using Machine Learning Techniques"

Endpoints:
- GET  /api/health            : Health check endpoint.
- POST /api/generate          : Generates new AI music given num_notes and temperature.
- GET  /api/download/<filename> : Securely downloads generated MIDI files from generated/ directory.
"""

import os
import time
import uuid
import logging
import threading
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

# Import the existing MusicGenerator class and seed helper
from music_generator import MusicGenerator, get_default_musical_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MusicApp")

# Initialize Flask application
app = Flask(__name__)

# Derive absolute project directories to ensure reliable path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "ai_music_model.keras")
PREPROCESSING_PATH = os.path.join(MODELS_DIR, "music_preprocessing.json")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")

# Retention management configuration: retain up to 100 most recently modified .mid files by default
DEFAULT_MAX_GENERATED_FILES = int(os.environ.get("MAX_GENERATED_FILES", 100))
app.config["MAX_GENERATED_FILES"] = DEFAULT_MAX_GENERATED_FILES

# Ensure the generated output folder exists
os.makedirs(GENERATED_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Thread-Safe Active Download Registry
# Protects in-flight served files from deletion across Windows & POSIX
# ---------------------------------------------------------------------
_active_downloads_lock = threading.Lock()
_active_download_counts = {}  # { filename: active_readers_count }


def acquire_active_download(filename):
    """Marks a filename as actively being served/downloaded."""
    with _active_downloads_lock:
        _active_download_counts[filename] = _active_download_counts.get(filename, 0) + 1


def release_active_download(filename):
    """Releases an active download reference for a filename."""
    with _active_downloads_lock:
        if filename in _active_download_counts:
            _active_download_counts[filename] -= 1
            if _active_download_counts[filename] <= 0:
                del _active_download_counts[filename]


def get_active_downloads():
    """Returns a copy set of currently active download filenames."""
    with _active_downloads_lock:
        return set(_active_download_counts.keys())


def is_download_active(filename):
    """Checks whether a filename is currently actively being downloaded."""
    with _active_downloads_lock:
        return filename in _active_download_counts and _active_download_counts[filename] > 0


def cleanup_generated_files(directory=None, max_files=None, exclude_filenames=None):
    """
    Safely enforces a retention policy on generated MIDI files.
    Retains the `max_files` most recently modified regular .mid files.

    Guarantees:
    - Only inspects regular files directly in `directory` ending with .mid.
    - Explicitly protects in-flight active downloads from deletion.
    - Never follows or deletes symlinks.
    - Never deletes subdirectories or non-.mid files.
    - Never deletes files outside `directory`.
    - Excludes `exclude_filenames` (e.g. the newly generated file).
    - Catches and logs all filesystem errors gracefully so generation never fails.
    """
    if directory is None:
        directory = GENERATED_DIR
    if max_files is None:
        max_files = app.config.get("MAX_GENERATED_FILES", DEFAULT_MAX_GENERATED_FILES)
    if exclude_filenames is None:
        exclude_filenames = set()
    elif isinstance(exclude_filenames, (list, tuple)):
        exclude_filenames = set(exclude_filenames)

    if max_files <= 0:
        return 0

    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return 0

    # Combine explicit exclusions with in-process active downloads
    active_served = get_active_downloads()
    protected_files = set(exclude_filenames).union(active_served)

    deleted_count = 0
    try:
        candidates = []
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    # Guard 1: Do not follow symlinks
                    if entry.is_symlink():
                        continue

                    # Guard 2: Must be a regular file directly in this folder
                    if not entry.is_file(follow_symlinks=False):
                        continue

                    # Guard 3: Must end with .mid (case-insensitive)
                    if not entry.name.lower().endswith(".mid"):
                        continue

                    # Guard 4: Exclude protected files (active downloads or currently generated file)
                    if entry.name in protected_files or is_download_active(entry.name):
                        continue

                    # Get modification time safely
                    stat_info = entry.stat(follow_symlinks=False)
                    candidates.append((stat_info.st_mtime, entry.name, entry.path))
                except OSError as e:
                    logger.warning("Error inspecting file %s: %s", entry.name, str(e))
                    continue

        # If total count is within limit, nothing to prune
        if len(candidates) <= max_files:
            return 0

        # Sort by st_mtime ascending (oldest first)
        candidates.sort(key=lambda x: x[0])

        # Number of files to prune
        excess_count = len(candidates) - max_files
        to_delete = candidates[:excess_count]

        for mtime, fname, fpath in to_delete:
            try:
                # Re-verify in-process active download right before deletion
                if is_download_active(fname):
                    logger.info("Preserving actively served MIDI file: %s", fname)
                    continue

                # Additional sanity check: ensure file is strictly within directory
                real_fpath = os.path.abspath(fpath)
                if os.path.dirname(real_fpath) != directory:
                    logger.warning("Refusing to delete file outside directory: %s", real_fpath)
                    continue

                if os.path.islink(real_fpath):
                    continue

                os.remove(real_fpath)
                deleted_count += 1
                logger.info("Retention cleanup pruned old MIDI: %s", fname)
            except OSError as e:
                logger.warning("Could not delete file %s during retention cleanup: %s", fname, str(e))

    except Exception as e:
        logger.error("Unexpected error during retention cleanup: %s", str(e))

    return deleted_count

# ---------------------------------------------------------------------
# Load Model and Engine ONCE on server startup
# ---------------------------------------------------------------------
logger.info("Initializing MusicGenerator engine and loading model...")
try:
    generator = MusicGenerator(
        model_path=MODEL_PATH,
        preprocessing_path=PREPROCESSING_PATH,
    )
    # Pre-cache a default valid musical seed for requests
    DEFAULT_SEED = get_default_musical_seed(generator)
    logger.info("MusicGenerator initialized successfully. Server ready.")
except Exception as e:
    logger.exception("Failed to initialize MusicGenerator during startup: %s", str(e))
    generator = None
    DEFAULT_SEED = None


# ---------------------------------------------------------------------
# Route 0: Web Dashboard Home Page
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """
    Renders the AI Music Composition Web Dashboard.
    """
    return render_template("index.html")


# ---------------------------------------------------------------------
# Route 1: Health Check
# ---------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    GET /api/health
    Returns server status, model readiness, and configuration metadata.
    """
    if generator is None:
        return jsonify({
            "status": "error",
            "message": "MusicGenerator engine failed to initialize on startup."
        }), 500

    return jsonify({
        "status": "healthy",
        "message": "AI Music Composition Server is running",
        "model_loaded": True,
        "sequence_length": generator.sequence_length,
        "unique_pitches_count": generator.num_pitches,
        "duration_scale": generator.duration_scale,
        "gap_scale": generator.gap_scale,
        "max_generated_files": app.config.get("MAX_GENERATED_FILES", DEFAULT_MAX_GENERATED_FILES)
    }), 200


# ---------------------------------------------------------------------
# Route 2: Generate Music
# ---------------------------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def generate_music():
    """
    POST /api/generate
    Accepts JSON body:
    {
        "num_notes": 50,       # Optional (integer: 10 to 500, default: 50)
        "temperature": 0.8     # Optional (float: 0.1 to 1.5, default: 0.8)
    }

    Generates music, saves it as a .mid file in generated/, and returns metadata.
    """
    if generator is None or DEFAULT_SEED is None:
        return jsonify({
            "status": "error",
            "message": "Model engine is not loaded. Please verify model files."
        }), 500

    # 0. Validate Request Body and Content-Type
    raw_body = request.get_data(cache=True).strip()
    if not raw_body:
        return jsonify({
            "status": "error",
            "message": "Request body cannot be empty. Please provide a JSON object (e.g., {})."
        }), 400

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Invalid Content-Type. Request must specify 'Content-Type: application/json'."
        }), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Malformed JSON payload in request body."
        }), 400

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "JSON payload must be an object (e.g., {\"num_notes\": 50, \"temperature\": 0.8})."
        }), 400

    # 1. Validate 'num_notes' (Allowed range: 10 to 500)
    raw_num_notes = data.get("num_notes", 50)
    try:
        num_notes = int(raw_num_notes)
    except (ValueError, TypeError):
        return jsonify({
            "status": "error",
            "message": f"Invalid 'num_notes' value: '{raw_num_notes}'. It must be an integer."
        }), 400

    if not (10 <= num_notes <= 500):
        return jsonify({
            "status": "error",
            "message": f"'num_notes' must be between 10 and 500. Received: {num_notes}"
        }), 400

    # 2. Validate 'temperature' (Allowed range: 0.1 to 1.5)
    raw_temperature = data.get("temperature", 0.8)
    try:
        temperature = float(raw_temperature)
    except (ValueError, TypeError):
        return jsonify({
            "status": "error",
            "message": f"Invalid 'temperature' value: '{raw_temperature}'. It must be a number."
        }), 400

    if not (0.1 <= temperature <= 1.5):
        return jsonify({
            "status": "error",
            "message": f"'temperature' must be between 0.1 and 1.5. Received: {temperature}"
        }), 400

    # 3. Generate Notes and MIDI
    try:
        start_time = time.time()

        # Generate notes using the pre-cached valid training seed
        notes = generator.generate_notes(
            seed_sequence=DEFAULT_SEED,
            num_notes=num_notes,
            temperature=temperature
        )

        # Unique filename using timestamp and short uuid
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]
        filename = f"music_{timestamp_str}_{unique_token}.mid"
        output_filepath = os.path.join(GENERATED_DIR, filename)

        # Save generated notes as standard MIDI file
        generator.save_to_midi(notes, output_file=output_filepath)

        # Safely enforce retention policy after successful MIDI generation
        try:
            cleanup_generated_files(
                directory=GENERATED_DIR,
                max_files=app.config.get("MAX_GENERATED_FILES", DEFAULT_MAX_GENERATED_FILES),
                exclude_filenames={filename}
            )
        except Exception as cleanup_err:
            logger.warning("Retention cleanup warning: %s", str(cleanup_err))

        elapsed_time = round(time.time() - start_time, 2)

        # Calculate total composition playback duration in seconds
        total_duration = round(sum(n["gap"] for n in notes), 2)

        logger.info(
            "Successfully generated %s (%d notes, temp=%.2f) in %.2fs",
            filename, num_notes, temperature, elapsed_time
        )

        return jsonify({
            "status": "success",
            "message": "Music generated successfully",
            "filename": filename,
            "download_url": f"/api/download/{filename}",
            "num_notes": num_notes,
            "temperature": temperature,
            "duration_seconds": total_duration,
            "generation_time_seconds": elapsed_time
        }), 201

    except Exception as e:
        logger.exception("Error occurred during music generation: %s", str(e))
        return jsonify({
            "status": "error",
            "message": f"Music generation failed: {str(e)}"
        }), 500


# ---------------------------------------------------------------------
# Route 3: Download Generated MIDI File
# ---------------------------------------------------------------------
@app.route("/api/download/<path:filename>", methods=["GET"])
def download_music(filename):
    """
    GET /api/download/<filename>
    Securely serves MIDI files from the generated/ folder.
    Guards against path traversal and directory escapes.
    """
    # Sanitize the filename to strip directory traversal tokens like '..' or '/'
    clean_filename = secure_filename(filename)

    # Check for empty or altered filename
    if not clean_filename or clean_filename != filename:
        return jsonify({
            "status": "error",
            "message": "Invalid filename or path traversal detected."
        }), 400

    # Ensure the file has a .mid extension
    if not clean_filename.lower().endswith(".mid"):
        return jsonify({
            "status": "error",
            "message": "Only MIDI files (.mid) can be downloaded."
        }), 400

    # Check if file exists inside the generated folder
    target_path = os.path.join(GENERATED_DIR, clean_filename)
    if not os.path.isfile(target_path):
        return jsonify({
            "status": "error",
            "message": f"File '{clean_filename}' was not found."
        }), 404

    # Acquire thread-safe active download reference
    acquire_active_download(clean_filename)
    try:
        response = send_from_directory(
            directory=GENERATED_DIR,
            path=clean_filename,
            as_attachment=True,
            mimetype="audio/midi",
            download_name=clean_filename
        )

        # Ensure release is called exactly once upon stream completion or disconnection
        released = False

        def safe_release():
            nonlocal released
            if not released:
                released = True
                release_active_download(clean_filename)

        # In Werkzeug, send_from_directory defaults direct_passthrough=True, which bypasses
        # response.close() during WSGI app iteration. Setting direct_passthrough=False ensures
        # Werkzeug wraps the stream in a ClosingIterator that reliably calls response.close().
        response.direct_passthrough = False
        response.call_on_close(safe_release)

        # Fail-safe: also wrap the underlying FileWrapper close method
        if hasattr(response, "response") and hasattr(response.response, "close"):
            orig_file_close = response.response.close

            def wrapped_file_close():
                try:
                    orig_file_close()
                finally:
                    safe_release()

            response.response.close = wrapped_file_close

        return response
    except Exception:
        # Guarantee release if response preparation fails
        release_active_download(clean_filename)
        raise


# ---------------------------------------------------------------------
# Local Development Server Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # In development mode, run on port 5000
    # Note: debug=False prevents reloading the TensorFlow model twice
    print("Starting AI Music Composition Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
