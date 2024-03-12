import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import cv2
import time
from my_utils import utils as u

# Model Load 및 Session
ort_session = ort.InferenceSession('model/best.ort')

# 모델 입력 정보
model_inputs = ort_session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
input_shape = model_inputs[0].shape


# 모델 출력 정보
model_output = ort_session.get_outputs()
output_names = [model_output[i].name for i in range(len(model_output))]

# 이미지 리사이징 >> 모델 입력 사이즈와 맞추기
image = cv2.imread('images/test3.jpg')
image_height, image_width = image.shape[:2]
input_height, input_width = input_shape[2:]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (input_width, input_height))

# 이미지 정규화
input_image = resized / 255.0 # 픽셀 값을 0과 1 사이로 정규화
input_image = input_image.transpose(2,0,1) # 이미지 차원을 변경. (높이, 너비, 채널) -> (채널, 높이, 너비)
input_tensor = input_image[np.newaxis, :,:,:].astype(np.float32) # 입력 텐서 생성 및 타입 변경

# 모델 로딩 시간 start
start = time.time()

# 모델 실행 및 결과 저장
outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
predictions = np.squeeze(outputs).T
print(predictions.shape)

# NMS 사용하기
# predictions = u.NMS(predictions)

## 객체 바운딩 박스
# 결과 조건 설정 (안해도 상관 없음..)
conf_thresold = 0.25
scores = np.max(predictions[:, 4:], axis=1)
predictions = predictions[scores > conf_thresold, :]
scores = scores[scores > conf_thresold]

# NMS algorithm >> 바운딩 박스 그리기
sorted_pred=predictions[np.argsort(predictions[:,4])]
sorted_pred[:,:4]=u.xywh2xyxy(sorted_pred[:,:4])

# # 바운딩 박스 시각화
# view=image[:,:,::-1].copy()
# for pred in sorted_pred:
#     x1, y1, x2, y2 = map(int, pred[:4])
#     view = cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), 1) # 바운딩 박스 시각화
#
# plt.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
# plt.show()

averages = []
for i in range(4, len(sorted_pred[0])):  # 인덱스 0부터 3까지는 좌표값이므로 제외
    total = sum(result[i] for result in sorted_pred)
    average = total / len(sorted_pred)
    averages.append(average)

# 평균이 가장 높은 리스트 선택
max_average_index = averages.index(max(averages))
selected_list = sorted_pred[max_average_index]
selected_list = list(selected_list)


classes = ['PAPERING_BREAK',
           'PAPERING_ERROR_COTTON',
           'PAPERING_ERROR_JOINT',
           'PAPERING_EXCITED',
           'PAPERING_MOLD',
           'PAPERING_POLLUTION',
           'PAPERING_UNDEVELOPED',
           'PAPERING_WRINKLE']

print("평균이 가장 높은 리스트:", selected_list)

data = {
    "0" : selected_list[0],
    "1" : selected_list[1],
    "2" : selected_list[2],
    "3" : selected_list[3],
    "acc" : round(max(selected_list[4:] * 100), 2),
    "class" : classes[selected_list.index(max(selected_list[4:])) - 4]
}

print(data)


# 모델 로딩 시간 start
end = time.time()

print("ort :: ", start - end)





