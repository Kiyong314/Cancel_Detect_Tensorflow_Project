# 4단계 필터링 기반 폐 분할 알고리즘

의료 영상에서 정확하고 안정적인 폐 영역 분할을 위한 개선된 알고리즘입니다.

## 🎯 주요 특징

- ✅ **양쪽 폐 보존** 보장
- ✅ **크기 불균형** 문제 해결
- ✅ **한쪽 폐 누락** 방지
- ✅ **의료 장비 제거** (침대, 테이블 등)
- ✅ **자동 임계값** 계산

## 📋 알고리즘 구조

```
원본 이미지 → 1차 필터 → 2차 필터 → 3차 필터 → 4차 필터 → 최종 결과
```

### 1차 필터링 (Primary Mask)
- 가중 평균 기반 자동 임계값 계산
- 동적 임계값 조정 (이미지 밝기 적응)
- 침대/의료장비 조기 제거
- 형태학적 노이즈 제거

### 2차 필터링 (Secondary Filtering)
- **위치 기반**: 중심에서 너무 먼 영역 제거
- **형태 기반**: 비정상적 가로세로 비율/원형도 제거
- **크기 기반**: 너무 작거나 큰 영역 제거
- **밝기 기반**: 의료장비로 추정되는 밝은 영역 제거

### 3차 필터링 (Tertiary Filtering) ⭐ 핵심 개선
- **양쪽 폐 시드 기반 확장**
- **다중 위치 시드 검색**
- **양쪽 폐 강제 복원 메커니즘**
- **크기 제한 강화** (35% 이상 영역 제거)

### 4차 필터링 (Final Result)
- 원본 이미지 × 3차 마스크
- 폐 영역만 보존, 배경 제거

## 🛠️ 설치 및 의존성

```bash
pip install opencv-python numpy scipy scikit-image matplotlib
```

### 필요한 라이브러리
```python
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure, segmentation
import matplotlib.pyplot as plt
import glob
```

## 🚀 사용법

### 기본 사용
```python
# 이미지 로드 및 전처리
img = cv2.imread('lung_image.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))
img_normalized = img / 255.0

# 4단계 필터링 실행
primary_mask, secondary_mask, tertiary_mask, final_result = improved_weighted_average_segmentation(img_normalized)
```

### 개선된 3차 필터 사용
```python
# 개선된 시드 기반 3차 필터링
tertiary_mask_improved = apply_tertiary_filtering_improved(secondary_mask)
final_result = img_normalized * tertiary_mask_improved
```

### 결과 시각화
```python
# 상세 분석 및 시각화
visualize_improved_preprocessing_v2('path/to/image.jpg')
```

## 📊 주요 함수

| 함수명 | 설명 |
|--------|------|
| `improved_weighted_average_segmentation()` | 메인 4단계 필터링 함수 |
| `apply_secondary_filtering()` | 2차 정교한 필터링 |
| `apply_tertiary_filtering_improved()` | 개선된 3차 시드 기반 필터링 |
| `apply_tertiary_filtering_original()` | 기존 방식 3차 필터링 |
| `find_bilateral_lung_seeds()` | 양쪽 폐 시드 검색 |
| `force_bilateral_recovery()` | 양쪽 폐 강제 복원 |
| `visualize_improved_preprocessing_v2()` | 결과 시각화 |

## 🔧 주요 개선사항

### 1. 양쪽 폐 보존 메커니즘
```python
def force_bilateral_recovery(original_mask, current_mask, h, w):
    """한쪽 폐가 누락된 경우 원본에서 복원"""
    left_area = count_pixels_in_left_half()
    right_area = count_pixels_in_right_half()
    
    if left_area < minimum_threshold:
        restore_left_lung_from_original()
    if right_area < minimum_threshold:
        restore_right_lung_from_original()
```

### 2. 다중 시드 검색
```python
# 여러 위치에서 시드 탐색
search_centers = [
    (center_x, center_y),           # 기본 중심
    (center_x, center_y - h//6),    # 위쪽
    (center_x, center_y + h//6),    # 아래쪽
    (center_x - w//8, center_y),    # 왼쪽
    (center_x + w//8, center_y),    # 오른쪽
]
```

### 3. 크기 제한 강화
```python
# 비정상적으로 큰 영역 제거
if area_ratio > 0.35:  # 전체의 35% 이상
    remove_oversized_region()
    
if width > w * 0.75 or height > h * 0.85:
    remove_oversized_region()
```

## 📈 성능 지표

각 단계별 추적 정보:
- **Coverage**: 각 마스크가 차지하는 면적 비율
- **Balance**: 좌우 폐의 균형도 (1.0 = 완벽한 균형)
- **Seed Count**: 발견된 시드 개수
- **Region Stats**: 영역별 크기, 위치, 형태 정보

## 🎨 시각화 결과

실행 시 다음 정보들이 출력됩니다:

```
=== 개선된 3차 필터 결과 ===
1st mask coverage: 15.2%
2nd mask coverage: 12.8%
3rd mask coverage: 11.4%
Final result mean intensity: 0.087

Found 3 left seeds, 2 right seeds
Backup: 2 left regions, 1 right regions
Left lung area: 14250 pixels (5.4%)
Right lung area: 15890 pixels (6.1%)
Total lung area: 30140 pixels (11.5%)
Lung balance ratio: 0.90 (1.0 = perfect balance)
```

## ⚠️ 주의사항

1. **입력 이미지**: 512x512 크기로 리사이즈 권장
2. **정규화**: 0-1 범위로 정규화 필수
3. **그레이스케일**: 흑백 이미지만 지원
4. **메모리**: 대용량 이미지 처리 시 메모리 사용량 주의

## 🔍 문제 해결

### 한쪽 폐가 누락되는 경우
```python
# 개선된 3차 필터 사용
tertiary_mask = apply_tertiary_filtering_improved(secondary_mask)
```

### 너무 큰 영역이 선택되는 경우
```python
# 크기 제한 조정
max_area_ratio = 0.25  # 기본값: 0.35
```

### 시드를 찾지 못하는 경우
```python
# fallback 방식 사용
tertiary_mask = apply_tertiary_filtering_fallback(secondary_mask)
```

## 📚 참고자료

- OpenCV Documentation
- scikit-image Documentation
- 의료 영상 분할 관련 논문들


