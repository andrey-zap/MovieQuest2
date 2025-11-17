import os
import random
import requests
import hashlib
import json
from flask import Flask, render_template, request, session, redirect, url_for, send_file
from image_processor import detect_and_blur_text, save_processed_image

app = Flask(__name__)
app.secret_key = "change_this_secret_key"

# TMDB API base URLs
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Ensure processed images directory exists
PROCESSED_IMAGES_DIR = os.path.join(app.static_folder, "processed_images")
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

# Leaderboard file
LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    """Load leaderboard from JSON file"""
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, 'r') as f:
            return json.load(f)
    return []

def save_leaderboard(leaderboard):
    """Save leaderboard to JSON file"""
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(leaderboard, f, indent=2)

def add_score(name, score):
    """Add a score to the leaderboard"""
    leaderboard = load_leaderboard()
    leaderboard.append({"name": name, "score": score})
    # Sort by score descending
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    save_leaderboard(leaderboard)

def get_top_scores(limit=5):
    """Get top N scores from leaderboard"""
    leaderboard = load_leaderboard()
    return leaderboard[:limit]


def get_movie_question():
    """
    Fetch 4 popular movies from TMDB,
    choose one as the poster question.
    Returns:
    {
        "poster_url": "...",
        "options": ["A", "B", "C", "D"],
        "answer_index": 1
    }
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("TMDB_API_KEY environment variable is not set")

    response = requests.get(
        f"{TMDB_BASE_URL}/movie/popular",
        params={"api_key": api_key, "language": "en-US", "page": 1},
        timeout=5
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    results = [m for m in results if m.get("poster_path")]

    if len(results) < 4:
        raise RuntimeError("Not enough movie results to generate a question")

    chosen = random.sample(results, 4)
    correct_movie = random.choice(chosen)

    poster_url = TMDB_IMAGE_BASE + correct_movie["poster_path"]
    options = [m["title"] for m in chosen]
    answer_index = options.index(correct_movie["title"])
    
    # Generate a unique ID for this poster
    poster_id = hashlib.md5(poster_url.encode()).hexdigest()

    return {
        "poster_url": poster_url,
        "poster_id": poster_id,
        "options": options,
        "answer_index": answer_index
    }


@app.route("/")
def home():
    return render_template("welcome.html")


@app.route("/start-game", methods=["POST"])
def start_game():
    """Initialize game with player name"""
    player_name = request.form.get("player_name", "").strip()
    if not player_name:
        return redirect(url_for("home"))
    
    # Initialize game session
    session.clear()
    session["player_name"] = player_name
    session["score"] = 0
    session["question_count"] = 0
    
    return redirect(url_for("movie_question"))


@app.route("/movie-question", methods=["GET"])
def movie_question():
    # Check if player has started the game
    if "player_name" not in session:
        return redirect(url_for("home"))
    
    # Check if game is over (10 questions completed)
    if session.get("question_count", 0) >= 10:
        return redirect(url_for("game_over"))
    
    # Generate new question if needed
    if "movie_q" not in session:
        session["movie_q"] = get_movie_question()

    q = session["movie_q"]
    
    return render_template(
        "movie_question.html",
        question=q,
        player_name=session["player_name"],
        score=session["score"],
        question_number=session["question_count"] + 1
    )


@app.route("/check-answer", methods=["POST"])
def check_answer():
    """This route is no longer used - answers are checked client-side"""
    return redirect(url_for("movie_question"))


@app.route("/movie-question/next")
def next_question():
    """Move to the next question after answering"""
    if "player_name" not in session:
        return redirect(url_for("home"))
    
    # Get answer result from query params (sent by JavaScript)
    is_correct = request.args.get("correct") == "true"
    
    # Update score if correct
    if is_correct:
        session["score"] = session.get("score", 0) + 10
    
    # Increment question count
    session["question_count"] = session.get("question_count", 0) + 1
    
    # Clear current question
    session.pop("movie_q", None)
    
    # Check if game is over
    if session["question_count"] >= 10:
        return redirect(url_for("game_over"))
    
    return redirect(url_for("movie_question"))


@app.route("/game-over")
def game_over():
    """Show final score and leaderboard"""
    if "player_name" not in session:
        return redirect(url_for("home"))
    
    player_name = session["player_name"]
    score = session.get("score", 0)
    
    # Add score to leaderboard
    add_score(player_name, score)
    
    # Get top scores
    leaderboard = get_top_scores(5)
    
    return render_template(
        "game_over.html",
        player_name=player_name,
        score=score,
        leaderboard=leaderboard
    )


@app.route("/play-again")
def play_again():
    """Restart game with same player name"""
    if "player_name" not in session:
        return redirect(url_for("home"))
    
    player_name = session["player_name"]
    session.clear()
    session["player_name"] = player_name
    session["score"] = 0
    session["question_count"] = 0
    
    return redirect(url_for("movie_question"))


@app.route("/processed-poster/<poster_id>")
def get_processed_poster(poster_id):
    """
    Serve a processed poster with text blurred.
    Uses caching to avoid re-processing the same image.
    """
    try:
        print(f"=== get_processed_poster called with poster_id: {poster_id}")
        print(f"=== Session exists: {'movie_q' in session}")
        if "movie_q" in session:
            print(f"=== Session poster_id: {session['movie_q'].get('poster_id')}")
            print(f"=== Session poster_url: {session['movie_q'].get('poster_url')}")
        
        # Check if processed image already exists in cache
        processed_path = os.path.join(PROCESSED_IMAGES_DIR, f"{poster_id}.jpg")
        print(f"=== Looking for cached image at: {processed_path}")
        
        if os.path.exists(processed_path):
            print(f"=== Found cached image!")
            return send_file(processed_path, mimetype='image/jpeg')
        
        # Get the original poster URL from session
        if "movie_q" in session:
            session_poster_id = session["movie_q"].get("poster_id")
            original_url = session["movie_q"].get("poster_url")
            
            print(f"=== Comparing IDs - requested: {poster_id}, session: {session_poster_id}")
            
            if session_poster_id == poster_id and original_url:
                print(f"=== Processing image from: {original_url}")
                processed_image = detect_and_blur_text(original_url, blur_strength=51)
                
                if processed_image is not None:
                    print(f"=== Image processed successfully, saving to cache")
                    save_processed_image(processed_image, processed_path)
                    return send_file(processed_path, mimetype='image/jpeg')
                else:
                    print(f"=== Image processing returned None")
            
            # If IDs don't match or processing failed, redirect to original
            print(f"=== Redirecting to original URL: {original_url}")
            return redirect(original_url)
        
        print(f"=== No session found, returning 404")
        return "Image not found", 404
        
    except Exception as e:
        print(f"=== Error in get_processed_poster: {e}")
        import traceback
        traceback.print_exc()
        # Return original image if available
        if "movie_q" in session:
            return redirect(session["movie_q"]["poster_url"])
        return "Error processing image", 500


if __name__ == "__main__":
    app.run(debug=True)
