# OpenCV를 활용한 lane detection

<https://github.com/user-attachments/assets/8ba3fb8c-9194-483d-80be-cbad6af56073>

OpenCV와 컴퓨터 비전 기술을 활용하여 실시간 차선 검출 시스템을 구현한 프로젝트입니다. Bird's Eye View 변환과 슬라이딩 윈도우 알고리즘을 통해 도로 영상에서 차선을 검출합니다.

- OpenCV를 이용한 실시간 이미지 처리
- HSV 색공간 기반 흰색/노란색 차선 검출
- Perspective Transform을 통한 Bird's Eye View 변환
- 슬라이딩 윈도우 알고리즘으로 정확한 차선 추적
- 차선 중심으로부터의 편차 계산

## 주요 기능

### 1. HSV 색공간 기반 차선 검출

```python
# 흰색과 노란색 차선을 구분하여 검출하는 HSV 임계값 설정
WHITE_LANE_HSV_LOWER = np.array([0, 0, 120])
WHITE_LANE_HSV_UPPER = np.array([179, 50, 255])
YELLOW_LANE_HSV_LOWER = np.array([15, 40, 120])
YELLOW_LANE_HSV_UPPER = np.array([35, 255, 255])
```

RGB 대신 HSV 색공간을 사용하여 조명 변화에 더 robust한 차선 검출을 구현했습니다.

### 2. Perspective Transform (Bird's Eye View)

차선의 곡률 측정과 추적 정확도 향상을 위해 Perspective Transform 기법을 사용하여 도로 영상의 원근 효과를 제거하고 차선을 평행선으로 변환했습니다.

```python
# 사다리꼴 형태의 도로 영역을 직사각형으로 변환
source_points = np.float32([[180, 387], [30, 472], [460, 380], [610, 472]])
destination_points = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
```

### 3. 슬라이딩 윈도우 알고리즘

히스토그램 분석으로 차선의 시작점을 찾고, 윈도우를 위로 이동시키며 차선을 연속적으로 추적하는 알고리즘을 구현했습니다.

- 각 윈도우에서 컨투어 검출 및 무게중심 계산
- 윈도우 내 충분한 픽셀이 검출되면 다음 윈도우 위치 업데이트
- 이전 프레임 정보를 활용한 안정적인 추적

## 결과 시각화

프로젝트에서는 두 가지 실시간 시각화를 제공합니다:

- 원본 영상에 검출된 차선 영역을 오버레이한 최종 결과
- 슬라이딩 윈도우 과정을 보여주는 Bird's Eye View

## 참고 자료

이 프로젝트는 다음 영상을 참고하여 학습하고 구현했습니다:
**"OpenCV Python Tutorial for Beginners - Real Time Lane Detection"**  
<https://www.youtube.com/watch?v=QkfVvktGyEs>
