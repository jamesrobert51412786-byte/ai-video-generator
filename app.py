"""
AI Video Storyboard Generator
-----------------------------
A beginner-friendly Streamlit app that:
  1. Turns a video idea into a 3-scene storyboard (OpenRouter text model)
  2. Records a voiceover MP3 for each scene (free edge-tts library)
  3. Generates a character reference image + 3 video clips (Fal.ai)

Install:
    pip install streamlit requests edge-tts fal-client

Run:
    streamlit run app.py
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor

import edge_tts
import fal_client
import requests
import streamlit as st

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3-8b-instruct:free"

AUDIO_DIR = "audio"    # audio/scene_1.mp3, scene_2.mp3, ...
IMAGE_DIR = "images"   # images/character_reference.png

# Fal.ai models. Model names change over time; check fal.ai/models if one fails.
IMAGE_MODEL = "fal-ai/flux/schnell"
VIDEO_MODEL = "fal-ai/kling-video/v2.1/standard/image-to-video"
CLIP_SECONDS = "5"
# Rough price per 5-second clip for the model above (from Fal's docs when this
# was written). Check fal.ai/pricing for current numbers.
CLIP_PRICE_ESTIMATE = 0.25

VOICES = {
    "Aria (US, female)": "en-US-AriaNeural",
    "Guy (US, male)": "en-US-GuyNeural",
    "Jenny (US, female)": "en-US-JennyNeural",
    "Sonia (UK, female)": "en-GB-SoniaNeural",
    "Ryan (UK, male)": "en-GB-RyanNeural",
}

SYSTEM_PROMPT = """You are a professional video storyboard writer.

The user will give you a video idea. Break it into exactly 3 distinct scenes
in chronological order (scene 1 = beginning, scene 2 = middle, scene 3 = end).

Respond with ONLY valid JSON, no extra text, no markdown fences, in this shape:
{
  "title": "Short video title",
  "scenes": [
    {
      "scene_number": 1,
      "scene_title": "Short scene title",
      "voiceover": "The exact narration spoken in this scene (2-3 sentences).",
      "visuals": "Specific visual description: setting, subjects, camera angle, lighting, colors, motion."
    },
    { "scene_number": 2, "scene_title": "...", "voiceover": "...", "visuals": "..." },
    { "scene_number": 3, "scene_title": "...", "voiceover": "...", "visuals": "..." }
  ]
}"""


# ----------------------------------------------------------------------
# Storyboard (text) functions
# ----------------------------------------------------------------------
def get_secret(name: str) -> str:
    """Look for a key in Streamlit secrets, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # No secrets file exists; that's fine.
    return os.environ.get(name, "")


def generate_storyboard(idea: str, api_key: str, model: str) -> dict:
    """Send the idea to OpenRouter and return the storyboard as a dict."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Video idea: {idea}"},
        ],
        "temperature": 0.7,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return parse_json(content)


def parse_json(text: str) -> dict:
    """Pull the JSON object out of the model's reply, even if it adds extra text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The model did not return JSON.")
    return json.loads(text[start : end + 1])


# ----------------------------------------------------------------------
# Voiceover (audio) functions
# ----------------------------------------------------------------------
async def _synthesize(text: str, voice: str, path: str) -> None:
    """edge-tts is async, so the actual speech work happens here."""
    await edge_tts.Communicate(text, voice).save(path)


def generate_voiceover(text: str, scene_number: int, voice: str) -> str:
    """Convert text to speech and save it as audio/scene_<number>.mp3."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, f"scene_{scene_number}.mp3")
    asyncio.run(_synthesize(text, voice, path))
    return path


def generate_all_voiceovers(storyboard: dict, voice: str) -> list:
    """Create an MP3 for every scene. Adds an 'audio_path' to each scene.

    One failed scene never stops the others. Returns a list of error messages
    (empty if everything worked).
    """
    errors = []
    for number, scene in enumerate(storyboard.get("scenes", []), start=1):
        text = (scene.get("voiceover") or "").strip()
        if not text:
            errors.append(f"Scene {number} has no voiceover text.")
            continue
        try:
            scene["audio_path"] = generate_voiceover(text, number, voice)
        except Exception as err:
            errors.append(f"Scene {number} voiceover failed: {err}")
    return errors


# ----------------------------------------------------------------------
# Video generation (Fal.ai) functions
# ----------------------------------------------------------------------
def generate_reference_image(visual_description: str) -> tuple:
    """Create ONE character reference image with Flux Schnell.

    Returns (image_url, local_path). The URL is what the video model needs;
    the local copy is saved so you can look at it or reuse it later.
    """
    prompt = (
        "Character reference image, single main character clearly visible, "
        "centered, full body, clean simple background, consistent design. "
        f"{visual_description}"
    )
    result = fal_client.subscribe(
        IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": "landscape_16_9",
            "num_images": 1,
            "output_format": "png",
        },
        client_timeout=120,
    )
    image_url = result["images"][0]["url"]

    # Save a local copy.
    os.makedirs(IMAGE_DIR, exist_ok=True)
    local_path = os.path.join(IMAGE_DIR, "character_reference.png")
    download = requests.get(image_url, timeout=60)
    download.raise_for_status()
    with open(local_path, "wb") as image_file:
        image_file.write(download.content)

    return image_url, local_path


def generate_video_clip(image_url: str, scene_prompt: str) -> str:
    """Animate the reference image into a short clip. Returns the video URL."""
    result = fal_client.subscribe(
        VIDEO_MODEL,
        arguments={
            "prompt": scene_prompt,
            "image_url": image_url,
            "duration": CLIP_SECONDS,
            "aspect_ratio": "16:9",
            "negative_prompt": "blur, distort, low quality, text, watermark",
            "cfg_scale": 0.5,
        },
        client_timeout=600,  # video generation can take a few minutes
    )
    return result["video"]["url"]


def generate_all_videos(storyboard: dict) -> dict:
    """Run the full video pipeline for a storyboard.

    1. Makes a character reference image from scene 1's visuals.
    2. Makes one video clip per scene (all 3 run at the same time to save waiting).

    Returns a dict:
        {
          "reference_url": "...",
          "reference_path": "images/character_reference.png",
          "video_urls": ["clip 1 url", "clip 2 url", "clip 3 url"],
        }
    """
    scenes = storyboard.get("scenes", [])
    if not scenes:
        raise ValueError("The storyboard has no scenes.")

    reference_url, reference_path = generate_reference_image(scenes[0].get("visuals", ""))

    prompts = [
        f"{scene.get('scene_title', '')}. {scene.get('visuals', '')}".strip()
        for scene in scenes
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(generate_video_clip, reference_url, p) for p in prompts]
        video_urls = [future.result() for future in futures]  # keeps scene order

    return {
        "reference_url": reference_url,
        "reference_path": reference_path,
        "video_urls": video_urls,
    }


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------
def show_storyboard(data: dict, video_urls: list = None) -> None:
    """Render the storyboard nicely on the page."""
    st.subheader(data.get("title", "Your Storyboard"))
    for number, scene in enumerate(data.get("scenes", []), start=1):
        title = scene.get("scene_title", "")
        with st.container(border=True):
            st.markdown(f"### Scene {number}: {title}")
            st.markdown("**🎙️ Voiceover**")
            st.write(scene.get("voiceover", ""))

            audio_path = scene.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")

            st.markdown("**🎬 Visuals**")
            st.write(scene.get("visuals", ""))

            if video_urls and number <= len(video_urls):
                st.video(video_urls[number - 1])


# ----------------------------------------------------------------------
# Page layout
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Storyboard Generator", page_icon="🎬", layout="centered")

st.title("🎬 AI Storyboard Generator")
st.caption("Type a video idea and get a 3-scene storyboard with voiceover audio and video clips.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenRouter API key",
        value=get_secret("OPENROUTER_API_KEY"),
        type="password",
        help="Get a free key at openrouter.ai/keys",
    )
    model = st.text_input(
        "Model",
        value=DEFAULT_MODEL,
        help="Free model names change over time. Browse openrouter.ai/models and filter by 'free'.",
    )
    voice_label = st.selectbox("Narrator voice", list(VOICES.keys()))
    voice = VOICES[voice_label]

    st.divider()
    fal_key = st.text_input(
        "Fal.ai API key",
        value=get_secret("FAL_KEY"),
        type="password",
        help="Get a key at fal.ai/dashboard/keys. Video generation uses paid credits.",
    )

idea = st.text_area(
    "Your video idea",
    placeholder="Example: A 30-second ad for a coffee shop that roasts beans in the mountains",
    height=120,
)

# --- Step 1: storyboard + voiceovers (free) ---------------------------------
if st.button("Generate Storyboard", type="primary"):
    if not api_key:
        st.error("Please add your OpenRouter API key in the sidebar.")
    elif not idea.strip():
        st.warning("Please type a video idea first.")
    else:
        with st.spinner("Writing your storyboard and recording voiceovers..."):
            try:
                storyboard = generate_storyboard(idea.strip(), api_key, model.strip())
                audio_errors = generate_all_voiceovers(storyboard, voice)
                st.session_state["storyboard"] = storyboard
                st.session_state["audio_errors"] = audio_errors
                # A new storyboard makes any old videos out of date.
                st.session_state.pop("videos", None)
            except requests.exceptions.HTTPError as err:
                code = err.response.status_code
                if code == 401:
                    st.error("Your API key was rejected. Double-check it in the sidebar.")
                elif code == 404:
                    st.error("That model wasn't found. It may have been removed. Try another free model name in the sidebar.")
                elif code == 429:
                    st.error("Rate limit hit (common with free models). Wait a minute and try again.")
                else:
                    st.error(f"OpenRouter returned an error ({code}): {err.response.text}")
            except requests.exceptions.RequestException as err:
                st.error(f"Network problem: {err}")
            except (ValueError, KeyError, json.JSONDecodeError):
                st.error("The model's reply wasn't in the expected format. Click the button again to retry.")

# --- Step 2: video clips (costs money, so it has its own button) ------------
if "storyboard" in st.session_state:
    st.divider()
    cost = CLIP_PRICE_ESTIMATE * len(st.session_state["storyboard"].get("scenes", []))
    st.caption(
        f"Video generation uses paid Fal.ai credits: roughly ${cost:.2f} for all "
        "3 clips. It can take a few minutes."
    )
    if st.button("Generate Video Clips"):
        if not fal_key:
            st.error("Please add your Fal.ai API key in the sidebar.")
        else:
            os.environ["FAL_KEY"] = fal_key  # the fal_client library reads it from here
            with st.spinner("Creating character image and video clips (this can take a few minutes)..."):
                try:
                    st.session_state["videos"] = generate_all_videos(st.session_state["storyboard"])
                except Exception as err:
                    st.error(f"Video generation failed: {err}")

# --- Results ----------------------------------------------------------------
if "storyboard" in st.session_state:
    for message in st.session_state.get("audio_errors", []):
        st.warning(message)

    videos = st.session_state.get("videos")
    if videos:
        st.subheader("Character reference")
        st.image(videos["reference_path"], caption="Used as the starting frame for every clip")

    show_storyboard(
        st.session_state["storyboard"],
        video_urls=videos["video_urls"] if videos else None,
    )
