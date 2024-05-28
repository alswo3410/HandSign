from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
seq_length = 15

model = load_model('model-5.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
recognized_letter = '?'
saved_letters = []

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global seq, action_seq, recognized_letter
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            img = cv2.flip(frame, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :], v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                    angle = np.degrees(angle)

                    d = np.concatenate([joint.flatten(), angle])
                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    if len(seq) < seq_length:
                        continue

                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                    y_pred = model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.9:
                        continue

                    action = actions[i_pred]
                    action_seq.append(action)

                    if len(action_seq) < 3:
                        continue

                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        recognized_letter = action
                        print(f"Recognized letter: {recognized_letter}")  # 로그 출력

            # 현재 인식된 알파벳을 이미지에 표시
            cv2.putText(img, f'letter: {recognized_letter.upper()}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_letter', methods=['POST'])
def save_letter():
    global recognized_letter, saved_letters
    if recognized_letter != '?':
        saved_letters.append(recognized_letter)
        return jsonify({'saved_letters': saved_letters})
    return jsonify({'saved_letters': saved_letters})

@app.route('/delete_letter', methods=['POST'])
def delete_letter():
    global saved_letters
    data = request.get_json()
    letter_to_delete = data.get('letter')
    if letter_to_delete in saved_letters:
        saved_letters.remove(letter_to_delete)
    return jsonify({'saved_letters': saved_letters})

if __name__ == "__main__":
    app.run(debug=True)
