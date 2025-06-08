이 코드는 4단계 필터링 기반 폐 분할 알고리즘을 구현한 것입니다. 각 단계별로 설명드리겠습니다.
📋 전체 구조
원본 이미지 → 1차 필터 → 2차 필터 → 3차 필터 → 4차 필터 → 최종 결과

🔍 1차 필터링 (Primary Mask)
목적: 기본적인 폐 영역 후보를 추출
주요 과정:

가중 평균 기반 임계값 계산: 히스토그램 분석으로 최적 임계값 자동 탐지
동적 임계값 조정: 이미지 밝기에 따라 임계값 보정
이진화: 폐(어두운 영역)와 배경(밝은 영역) 분리
침대/테이블 조기 제거: 하단 영역의 의료 장비 제거
형태학적 정제: 작은 노이즈 제거, 구멍 메우기

python# 핵심 아이디어
optimal_threshold = find_best_threshold_using_variance()
binary_mask = apply_threshold(image, optimal_threshold)
cleaned_mask = remove_noise_and_bed_regions(binary_mask)

🎯 2차 필터링 (Secondary Filtering)
목적: 폐가 아닌 영역들을 정교하게 제거
필터링 조건들:
위치 기반 필터링

이미지 중심에서 너무 멀리 떨어진 영역 제거
경계 근처의 작은 영역들 제거

형태 기반 필터링
python# 세로/가로 비율 검사
aspect_ratio = height / width
if aspect_ratio > 4.0 or aspect_ratio < 0.2:
    remove_region()  # 너무 길거나 납작한 영역 제거

# 원형도 검사
circularity = 4 * π * area / perimeter²
if circularity < 0.05:
    remove_region()  # 너무 복잡한 모양 제거
크기 기반 필터링

너무 작은 영역 (< 0.1% of image) 제거
너무 큰 영역 (> 40% of image) 제거

밝기 기반 필터링

너무 밝은 영역 (평균 밝기 > 85%) 제거
침대나 의료 장비로 추정되는 영역 제거


🚀 3차 필터링 (Tertiary Filtering) - 핵심 개선
두 가지 방식 제공:
방식 1: 기존 방식 (Original)
pythondef apply_tertiary_filtering_original(mask):
    # 가장 큰 영역 1-2개만 선택
    regions = find_connected_components(mask)
    select_largest_regions(regions, max_count=2)
방식 2: 개선된 방식 (Improved) ⭐
pythondef apply_tertiary_filtering_improved(mask):
    # 1. 양쪽 폐 시드 찾기
    left_seeds = find_lung_seeds(left_half_of_mask)
    right_seeds = find_lung_seeds(right_half_of_mask)
    
    # 2. 시드 기반 영역 확장
    for seed in seeds:
        expanded_region = region_growing(mask, seed)
        
    # 3. 양쪽 폐 강제 복원
    ensure_both_lungs_present()
개선된 방식의 장점:

✅ 양쪽 폐 모두 보존
✅ 크기 불균형 문제 해결
✅ 한쪽 폐 누락 방지


🎨 4차 필터링 (Final Result)
목적: 최종 결과 생성
pythonfinal_result = original_image × tertiary_mask
원본 이미지에 3차 마스크를 곱해서 폐 영역만 남기고 나머지는 검은색으로 처리

🔧 주요 개선 사항
1. 양쪽 폐 보존 메커니즘
pythondef force_bilateral_recovery(original_mask, current_mask, h, w):
    left_area = count_pixels_in_left_half()
    right_area = count_pixels_in_right_half()
    
    if left_area < minimum_threshold:
        restore_left_lung_from_original()
    if right_area < minimum_threshold:
        restore_right_lung_from_original()
2. 다중 시드 검색
python# 여러 위치에서 시드 탐색
search_centers = [
    (center_x, center_y),           # 기본 중심
    (center_x, center_y - h//6),    # 위쪽
    (center_x, center_y + h//6),    # 아래쪽
    # ...
]
3. 크기 제한 강화
python# 비정상적으로 큰 영역 제거
if area_ratio > 0.35:  # 전체의 35% 이상
    remove_oversized_region()
    
if width > w * 0.75 or height > h * 0.85:
    remove_oversized_region()

📊 결과 검증
각 단계별로 다음 정보들을 추적:

Coverage: 각 마스크가 차지하는 면적 비율
Balance: 좌우 폐의 균형도
Seed Count: 발견된 시드 개수
Region Stats: 영역별 크기, 위치, 형태 정보

이러한 4단계 필터링을 통해 의료 영상에서 정확하고 안정적인 폐 분할을 수행할 수 있습니다! 🫁재시도K간단하게 ppt에 넣을 문구로 작성해줘편집4단계 필터링 기반 폐 분할 알고리즘
📋 전체 프로세스
원본 이미지 → 1차 필터 → 2차 필터 → 3차 필터 → 4차 필터 → 최종 결과

🔍 1차 필터링 (Primary Mask)

목적: 기본 폐 영역 후보 추출
주요 기능:

가중 평균 기반 자동 임계값 계산
동적 임계값 조정 (이미지 밝기 적응)
침대/의료장비 조기 제거
형태학적 노이즈 제거




🎯 2차 필터링 (Secondary Filtering)

목적: 정교한 비폐 영역 제거
필터링 조건:

위치 기반: 중심에서 너무 먼 영역 제거
형태 기반: 비정상적 가로세로 비율/원형도 제거
크기 기반: 너무 작거나 큰 영역 제거
밝기 기반: 의료장비로 추정되는 밝은 영역 제거




🚀 3차 필터링 (Tertiary Filtering) - 핵심 개선
기존 방식

가장 큰 1-2개 영역만 선택
문제점: 한쪽 폐 누락 가능성

개선된 방식 ⭐

양쪽 폐 시드 기반 확장
다중 위치 시드 검색
양쪽 폐 강제 복원 메커니즘
크기 제한 강화 (35% 이상 영역 제거)


🎨 4차 필터링 (Final Result)

기능: 원본 이미지 × 3차 마스크
결과: 폐 영역만 보존, 배경 제거


✅ 주요 개선사항

🫁 양쪽 폐 보존 보장
⚖️ 크기 불균형 문제 해결
🎯 누락 방지 안전장치
📏 비정상 크기 영역 제거 강화
