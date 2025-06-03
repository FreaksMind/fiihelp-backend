import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment


from ai import get_answer, sr, speak
from ai import get_answer

app = Flask(__name__)
CORS(app)


@app.route('/api/post_question', methods=['POST'])
def post_question():
    data = request.get_json()
    question = data.get("question")
    session_id = data.get("session_id")

    if not question:
        return jsonify({"status": "error", "message": "question missing"}), 400

    try:
        answer = get_answer(question, session_id)
        return jsonify({"answer": answer})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/audio_question', methods=['POST'])
def audio_question():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']
    try:
        # save audio to a temporary .webm file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            webm_path = tmp.name
            audio_file.save(webm_path)
        # .webm to .wav
        wav_path = webm_path.replace(".webm", ".wav")
        sound = AudioSegment.from_file(webm_path, format="webm")
        sound.export(wav_path, format="wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            question = recognizer.recognize_google(audio_data)
        print("Recognized question:", question)
        answer = get_answer(question)
        return jsonify({"answer": answer})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(webm_path):
                os.remove(webm_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass

if __name__ == '__main__':
    app.run(debug=True)
