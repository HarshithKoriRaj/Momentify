This project is a real-time, multimodal “happy moment” detector that automatically records short video clips when there is consensus that the user is happy across:
Facial expression (HSEmotion + InsightFace)
Voice tone (Wav2Vec2 emotion classifier)
Spoken text (Whisper ASR + Groq/Phidata text agent)
Vision-Language reasoning (Moondream2 VLM)
When at least 2 points of “Happy” votes are reached (with VLM counting as 2 points by itself), the system records a 10-second smart video to the smart_videos/ folder.
