import cv2
import numpy as np
import warnings
import threading
import torch
import time
import os
import pyaudio
from insightface.app import FaceAnalysis
from hsemotion.facial_emotions import HSEmotionRecognizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PIL import Image

# --- PHIDATA IMPORTS ---
try:
    from phi.agent import Agent
    from phi.model.groq import Groq
    PHIDATA_AVAILABLE = True
except ImportError:
    PHIDATA_AVAILABLE = False
    print("WARNING: Phidata/Groq not found. Install with 'pip install phidata groq'")

# --- 1. CONFIGURATION ---
warnings.filterwarnings("ignore")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"System: Running Video AI on {DEVICE}")

if not os.path.exists("smart_videos"):
    os.makedirs("smart_videos")

# API KEY CHECK
if "GROQ_API_KEY" not in os.environ:
    print("⚠️ WARNING: GROQ_API_KEY not found in environment variables.")
    print("   Text emotion analysis will be disabled.")

# --- 2. LOAD VISION MODELS ---
print("1. Loading Face Detection...")
app = FaceAnalysis(name="buffalo_l", providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print("2. Loading Fast Emotion (HSEmotion)...")
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu') 

print("3. Loading Moondream2 (Visual Reasoning)...")
try:
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26" 
    vlm_model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        low_cpu_mem_usage=False, device_map=None, 
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    print("   Moondream2 Loaded.")
except Exception as e:
    print(f"Error loading Moondream2: {e}")
    exit()

# --- 3. LOAD AUDIO/TEXT MODELS ---
print("4. Loading Audio Emotion (Wav2Vec2)...")
try:
    audio_classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=-1)
except Exception as e:
    audio_classifier = None

print("5. Loading Speech-to-Text (Whisper)...")
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=-1)
except Exception as e:
    asr_pipeline = None

# --- 4. INIT PHIDATA AGENT ---
text_agent = None
if PHIDATA_AVAILABLE and os.getenv("GROQ_API_KEY"):
    print("6. Initializing Phidata Agent (Groq)...")
    text_agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile"), 
        description="You are an emotion classifier. You output ONLY one word.",
        instructions=["Analyze the user text.", "Output only one of: Happy, Sad, Angry, Neutral, Surprise, Fear.", "Do not output sentences."]
    )

# --- 5. SHARED VARIABLES ---
# Vision
current_vlm_result = "VLM: Idle"
vlm_is_busy = False 
vlm_lock = threading.Lock() 

# Audio/Text
current_voice_emotion = "Waiting..."
current_transcription = "..."
current_text_emotion = "Waiting..." # From Phidata
audio_score = 0.0
audio_lock = threading.Lock()

# Recording
is_recording = False
video_writer = None
recording_start_time = 0
VIDEO_DURATION = 10.0  
last_trigger_time = 0  
COOLDOWN_SECONDS = 5.0 

# --- 6. THREAD: VLM (MOONDREAM) ---
def run_moondream2_thread(pil_image, fast_emotion, spoken_context):
    global current_vlm_result, vlm_is_busy
    try:
        # UPDATED PROMPT: Includes spoken text for context
        prompt = (
            f"The user said: \"{spoken_context}\". "
            f"The fast model saw '{fast_emotion}'. "
            f"Look at the face and consider the text. Is the person truly Happy? "
            f"Answer 'Yes' if happy, or provide the correct emotion."
        )
        enc_image = vlm_model.encode_image(pil_image)
        answer = vlm_model.answer_question(enc_image, prompt, tokenizer)
        with vlm_lock:
            current_vlm_result = f"VLM: {answer}"
    except Exception as e:
        print(f"VLM Error: {e}")
    finally:
        vlm_is_busy = False

# --- 7. THREAD: TEXT ANALYSIS (PHIDATA) ---
def analyze_text_emotion(text):
    global current_text_emotion
    if not text_agent: return
    try:
        response = text_agent.run(f"Identify the emotion behind this text: '{text}'")
        result = response.content.strip().title() 
        with audio_lock:
            current_text_emotion = result
            print(f"🧠 Phidata Analysis: '{text}' -> {result}")
    except Exception as e:
        print(f"Agent Error: {e}")

# --- 8. THREAD: AUDIO LISTENER ---
def run_audio_thread():
    global current_voice_emotion, audio_score, current_transcription
    
    if audio_classifier is None: return

    CHUNK_DURATION = 4 
    RATE = 16000
    CHUNK_SIZE = 1024
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK_SIZE * CHUNK_DURATION)):
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                except: break
            
            if not frames: continue
            audio_input = np.concatenate(frames)
            
            # 1. Detect Voice Tone
            emotion_results = audio_classifier(audio_input, top_k=1)
            voice_emotion = emotion_results[0]['label'].title() 
            score = emotion_results[0]['score']
            
            # 2. Transcribe Text
            transcribed_text = ""
            if asr_pipeline:
                try:
                    asr_result = asr_pipeline({"sampling_rate": RATE, "raw": audio_input})
                    transcribed_text = asr_result.get("text", "").strip()
                except: pass

            # 3. Update Variables
            with audio_lock:
                current_voice_emotion = voice_emotion
                audio_score = score
                if transcribed_text and len(transcribed_text) > 2:
                    current_transcription = transcribed_text
            
            # 4. Trigger Text Analysis
            if transcribed_text and len(transcribed_text) > 2:
                threading.Thread(target=analyze_text_emotion, args=(transcribed_text,)).start()

    except Exception as e:
        print(f"Audio Thread Error: {e}")
    finally:
        p.terminate()

audio_thread = threading.Thread(target=run_audio_thread, daemon=True)
audio_thread.start()

# --- 9. MAIN LOOP ---
cap = cv2.VideoCapture(0)
print("\n--- CONSENSUS AI STARTED ---")
print("Need 2 Points to record.")
print("Standard Models = 1 Point")
print("VLM (Moondream) = 2 Points (Super Vote)")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    height, width = frame.shape[:2]
    ai_frame = frame.copy()
    faces = app.get(ai_frame)

    # READ SHARED STATE
    with audio_lock:
        voice_emo = current_voice_emotion 
        text_emo = current_text_emotion   
        transcript = current_transcription

    # DRAW DASHBOARD
    cv2.rectangle(frame, (10, 10), (400, 130), (40, 40, 40), -1)
    cv2.putText(frame, "VOTING SYSTEM:", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    color_voice = (0, 255, 0) if "Happy" in voice_emo else (0, 0, 255)
    cv2.putText(frame, f"Voice: {voice_emo}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_voice, 2)
    
    color_text = (0, 255, 0) if "Happy" in text_emo else (0, 0, 255)
    cv2.putText(frame, f"Text:  {text_emo}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
    
    cv2.putText(frame, f"Said: \"{transcript[:25]}...\"", (20, 115), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

    happy_face_detected = False

    for face in faces:
        box = face.bbox.astype(int)
        face_img = ai_frame[box[1]:box[3], box[0]:box[2]]
        
        if face_img.size > 0:
            dom_emo, scores = fer.predict_emotions(face_img, logits=False)
            conf = max(scores)
            
            if dom_emo == 'Happiness' and conf > 0.65:
                happy_face_detected = True

            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"Vis: {dom_emo}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # TRIGGER VLM: Run more often (if conf < 0.80) so it can weigh in
            if conf < 0.80 and not vlm_is_busy:
                vlm_is_busy = True 
                pil_face = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                # PASS TRANSCRIPTION TO VLM
                threading.Thread(target=run_moondream2_thread, args=(pil_face, dom_emo, transcript)).start()
            
            # Show VLM Opinion
            with vlm_lock:
                if "Idle" not in current_vlm_result:
                    cv2.putText(frame, current_vlm_result[:40], (box[0], box[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- UPDATED VOTING LOGIC ---
    current_time = time.time()
    
    vote_visual = 1 if happy_face_detected else 0
    vote_voice = 1 if "happ" in voice_emo.lower() else 0
    vote_text = 1 if "happ" in text_emo.lower() or "joy" in text_emo.lower() else 0
    
    # VLM Weighted Vote (Worth 2 points)
    vote_vlm = 0
    with vlm_lock:
        if "Yes" in current_vlm_result or "Happy" in current_vlm_result:
            vote_vlm = 2
    
    total_votes = vote_visual + vote_voice + vote_text + vote_vlm
    
    # Show Votes
    cv2.putText(frame, f"Points: {total_votes}", (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Trigger: Needs 2 points (VLM alone can trigger, or 2 standard models)
    if total_votes >= 2 and not is_recording and (current_time - last_trigger_time > COOLDOWN_SECONDS):
        is_recording = True
        recording_start_time = current_time
        filename = f"smart_videos/consensus_happy_{int(current_time)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        print(f"🎥 Consensus Reached ({total_votes} pts)! Recording...")

    if is_recording:
        if video_writer is not None:
            video_writer.write(ai_frame)
            elapsed = current_time - recording_start_time
            remaining = max(0, int(VIDEO_DURATION - elapsed))
            
            if int(elapsed * 4) % 2 == 0: 
                cv2.circle(frame, (width - 50, 50), 15, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {remaining}s", (width - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if elapsed >= VIDEO_DURATION:
                is_recording = False
                video_writer.release()
                video_writer = None
                last_trigger_time = current_time
                print("⏹️ Video saved.")

    cv2.imshow("Consensus AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()