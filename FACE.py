import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 載入模型
model = load_model('emotion_recognition_model.h5')

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 迴圈讀取影像
while True:
    # 讀取影像
    ret, frame = cap.read()

    # 縮小影像
    frame = cv2.resize(frame, (224, 224))

    # 轉換影像格式
    frame = np.expand_dims(frame, axis=0)

    # 進行預測
    predictions = model.predict(frame)

    # 顯示預測結果
    for i, emotion in enumerate(predictions[0]):
        cv2.putText(frame, emotion, (10, 20 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 顯示影像
    cv2.imshow('Emotion Recognition', frame)

    # 按下 q 鍵結束程式
    if cv2.waitKey(1) == ord('q'):
        break

# 關閉攝影機
cap.release()

# 關閉視窗
cv2.destroyAllWindows()
