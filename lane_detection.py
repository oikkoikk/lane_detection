import cv2
import numpy as np
import socket
import struct

# TCP 서버 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("127.0.0.1", 65432))
server_socket.listen(1)
print("Python 서버 시작. Unity 연결 대기 중...")

client_socket, addr = server_socket.accept()
print(f"Unity 클라이언트 연결됨: {addr}")


# 차선 검출을 위한 고정 HSV 임계값 설정
# 흰색 차선용 HSV 범위
WHITE_LANE_HSV_LOWER = np.array([0, 0, 120])
WHITE_LANE_HSV_UPPER = np.array([179, 50, 255])

# 노란색 차선용 HSV 범위
YELLOW_LANE_HSV_LOWER = np.array([15, 40, 120])
YELLOW_LANE_HSV_UPPER = np.array([35, 255, 255])

# 슬라이딩 윈도우 설정
WINDOW_HEIGHT = 40  # 윈도우 높이
WINDOW_MARGIN = 50  # 윈도우 좌우 마진

# 이전 프레임의 차선 좌표 저장용
previous_left_lane_x_coords = []
previous_right_lane_x_coords = []


def process_lane_detection(image):
    """차선 검출 함수"""
    global previous_left_lane_x_coords, previous_right_lane_x_coords

    # 이미지 크기를 640x480으로 조정
    resized_frame = cv2.resize(image, (640, 480))

    # 원근 변환을 위한 기준점 설정
    perspective_top_left = (180, 387)
    perspective_bottom_left = (30, 472)
    perspective_top_right = (460, 380)
    perspective_bottom_right = (610, 472)

    # 기준점들을 원본 이미지에 표시
    cv2.circle(resized_frame, perspective_top_left, 5, (0, 0, 255), -1)
    cv2.circle(resized_frame, perspective_bottom_left, 5, (0, 0, 255), -1)
    cv2.circle(resized_frame, perspective_top_right, 5, (0, 0, 255), -1)
    cv2.circle(resized_frame, perspective_bottom_right, 5, (0, 0, 255), -1)

    # 원근 변환 실행 (Bird's Eye View로 변환)
    source_points = np.float32(
        [
            perspective_top_left,
            perspective_bottom_left,
            perspective_top_right,
            perspective_bottom_right,
        ]
    )
    destination_points = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # 변환 행렬 생성 및 이미지 변환
    transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    birds_eye_view_frame = cv2.warpPerspective(
        resized_frame, transform_matrix, (640, 480)
    )

    # HSV 색공간으로 변환하여 차선 검출
    hsv_frame = cv2.cvtColor(birds_eye_view_frame, cv2.COLOR_BGR2HSV)

    # 전처리: 가우시안 블러로 노이즈 제거
    blurred_frame = cv2.GaussianBlur(birds_eye_view_frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # 흰색 차선을 위한 HSV 마스크
    white_mask = cv2.inRange(hsv_frame, WHITE_LANE_HSV_LOWER, WHITE_LANE_HSV_UPPER)

    # 노란색 차선을 위한 HSV 마스크
    yellow_mask = cv2.inRange(hsv_frame, YELLOW_LANE_HSV_LOWER, YELLOW_LANE_HSV_UPPER)

    # 흰색과 노란색 마스크를 결합
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

    # 히스토그램을 이용한 차선 시작점 찾기
    bottom_half_histogram = np.sum(lane_mask[lane_mask.shape[0] // 2 :, :], axis=0)
    image_midpoint = int(bottom_half_histogram.shape[0] / 2)
    left_lane_base_x = np.argmax(bottom_half_histogram[:image_midpoint])
    right_lane_base_x = (
        np.argmax(bottom_half_histogram[image_midpoint:]) + image_midpoint
    )

    # 슬라이딩 윈도우를 위한 변수 초기화
    current_y = 472
    left_lane_x_coords = []
    right_lane_x_coords = []

    # 윈도우 표시용 마스크 복사본
    sliding_window_mask = lane_mask.copy()

    # 슬라이딩 윈도우로 차선 추적
    while current_y > 0:
        # 좌측 차선 검출 영역 설정
        left_window = lane_mask[
            current_y - WINDOW_HEIGHT : current_y, left_lane_base_x - WINDOW_MARGIN : left_lane_base_x + WINDOW_MARGIN
        ]
        left_contours, _ = cv2.findContours(
            left_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in left_contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
                left_lane_x_coords.append(left_lane_base_x - WINDOW_MARGIN + centroid_x)
                left_lane_base_x = left_lane_base_x - WINDOW_MARGIN + centroid_x

        # 우측 차선 검출 영역 설정
        right_window = lane_mask[
            current_y - WINDOW_HEIGHT : current_y, right_lane_base_x - WINDOW_MARGIN : right_lane_base_x + WINDOW_MARGIN
        ]
        right_contours, _ = cv2.findContours(
            right_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in right_contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
                right_lane_x_coords.append(right_lane_base_x - WINDOW_MARGIN + centroid_x)
                right_lane_base_x = right_lane_base_x - WINDOW_MARGIN + centroid_x

        # 슬라이딩 윈도우 사각형 그리기
        cv2.rectangle(
            sliding_window_mask,
            (left_lane_base_x - WINDOW_MARGIN, current_y),
            (left_lane_base_x + WINDOW_MARGIN, current_y - WINDOW_HEIGHT),
            (255, 255, 255),
            2,
        )
        cv2.rectangle(
            sliding_window_mask,
            (right_lane_base_x - WINDOW_MARGIN, current_y),
            (right_lane_base_x + WINDOW_MARGIN, current_y - WINDOW_HEIGHT),
            (255, 255, 255),
            2,
        )
        current_y -= WINDOW_HEIGHT

    # 차선이 검출되지 않은 경우 이전 프레임 데이터 사용
    if len(left_lane_x_coords) == 0:
        left_lane_x_coords = previous_left_lane_x_coords
    else:
        previous_left_lane_x_coords = left_lane_x_coords

    if len(right_lane_x_coords) == 0:
        right_lane_x_coords = previous_right_lane_x_coords
    else:
        previous_right_lane_x_coords = right_lane_x_coords

    # 차선 중심 계산 및 편차 계산
    lane_detected = len(left_lane_x_coords) > 0 and len(right_lane_x_coords) > 0
    steering_deviation = 0.0
    detection_confidence = 0.0

    if lane_detected:
        # 차선 중심점 계산
        image_center_x = 320  # 이미지 중심
        if len(left_lane_x_coords) > 0 and len(right_lane_x_coords) > 0:
            lane_center_x = (
                left_lane_x_coords[-1] + right_lane_x_coords[-1]
            ) / 2  # 가장 가까운 차선의 중심
            steering_deviation = (
                lane_center_x - image_center_x
            ) / image_center_x  # 정규화된 편차
            detection_confidence = (
                min(len(left_lane_x_coords), len(right_lane_x_coords)) / 12.0
            )  # 검출된 포인트 수 기반 신뢰도

    # 차선 영역 하이라이트를 위한 다각형 생성
    min_lane_points = (
        min(len(left_lane_x_coords), len(right_lane_x_coords)) if lane_detected else 0
    )

    if min_lane_points > 0:
        # 차선 영역을 나타내는 사각형 좌표 생성
        lane_quad_top_left = (left_lane_x_coords[0], 472)
        lane_quad_bottom_left = (left_lane_x_coords[min_lane_points - 1], 0)
        lane_quad_top_right = (right_lane_x_coords[0], 472)
        lane_quad_bottom_right = (right_lane_x_coords[min_lane_points - 1], 0)

        # 다각형 포인트 배열 생성
        lane_polygon_points = np.array(
            [
                lane_quad_top_left,
                lane_quad_bottom_left,
                lane_quad_bottom_right,
                lane_quad_top_right,
            ],
            dtype=np.int32,
        )
        lane_polygon_points = lane_polygon_points.reshape((-1, 1, 2))

        # 차선 영역 하이라이트를 위한 오버레이 생성
        lane_overlay = birds_eye_view_frame.copy()
        cv2.fillPoly(lane_overlay, [lane_polygon_points], (0, 255, 0))

        # 투명도를 적용한 차선 영역 표시
        overlay_alpha = 0.2
        cv2.addWeighted(
            lane_overlay,
            overlay_alpha,
            birds_eye_view_frame,
            1 - overlay_alpha,
            0,
            birds_eye_view_frame,
        )

        # 원본 시점으로 역변환하여 차선 영역 표시
        inverse_transform_matrix = cv2.getPerspectiveTransform(
            destination_points, source_points
        )
        original_view_lane_image = cv2.warpPerspective(
            birds_eye_view_frame, inverse_transform_matrix, (640, 480)
        )

        # 원본 이미지와 차선 영역 합성
        final_result = cv2.addWeighted(
            resized_frame, 1, original_view_lane_image, 0.5, 0
        )
        cv2.imshow("Lane Detection - Final Result", final_result)

    # 슬라이딩 윈도우 결과 표시
    cv2.imshow("Lane Detection - Sliding Windows", sliding_window_mask)

    return steering_deviation, lane_detected, detection_confidence


# Unity에서 이미지를 받아 처리하는 메인 루프
try:
    while True:
        # Unity에서 이미지 크기 수신
        image_size_data = client_socket.recv(4)
        if not image_size_data:
            break

        received_image_size = struct.unpack("I", image_size_data)[0]

        # 이미지 데이터 수신
        received_image_data = b""
        while len(received_image_data) < received_image_size:
            data_chunk = client_socket.recv(
                received_image_size - len(received_image_data)
            )
            if not data_chunk:
                break
            received_image_data += data_chunk

        # JPG 데이터를 OpenCV 이미지로 디코딩
        image_array = np.frombuffer(received_image_data, np.uint8)
        received_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if received_image is not None:
            # 차선 검출 처리 실행
            steering_deviation, lane_detected, detection_confidence = (
                process_lane_detection(received_image)
            )

            # Unity로 결과 데이터 전송
            result_packet = struct.pack(
                "fff",
                steering_deviation,
                1.0 if lane_detected else 0.0,
                detection_confidence,
            )
            client_socket.send(result_packet)

            print(
                f"조향 편차: {steering_deviation:.3f}, 차선 검출: {lane_detected}, 신뢰도: {detection_confidence:.3f}"
            )

        # ESC 키 입력 시 프로그램 종료
        if cv2.waitKey(1) == 27:
            break

except Exception as error:
    print(f"오류 발생: {error}")
finally:
    # 리소스 정리
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
