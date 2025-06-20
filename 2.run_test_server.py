#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FocusNet-LC 웹 인터페이스 - 폐암 진단, Grad-CAM 시각화, 의사 진단, 예약 및 개인화된 건강 권고
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import io
import base64
from PIL import Image
from scipy import ndimage
from skimage import morphology, measure, segmentation
from lungmask import LMInferer
import SimpleITK as sitk
from datetime import datetime, timedelta
import json
from tensorflow.keras.models import model_from_json
h5_path = "lung_model.h5"

# 설정
IMG_SIZE = 512
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 업로드 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 클래스 라벨
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
CLASS_COLORS = ['#28a745', '#ffc107', '#dc3545']  # 초록, 노랑, 빨강

def generate_health_recommendations(prediction_result, doctor_diagnosis, survey_data):
    """
    AI 예측 결과, 의사 진단, 설문조사 결과에 따른 개인화된 건강 권고사항 생성
    
    Args:
        prediction_result: AI 예측 결과 딕셔너리
        doctor_diagnosis: 의사 진단 딕셔너리
        survey_data: 설문조사 데이터 딕셔너리
    
    Returns:
        recommendations: 건강 권고사항 딕셔너리
    """
    
    ai_prediction = prediction_result['prediction']
    ai_confidence = prediction_result['confidence']
    doctor_final_diagnosis = doctor_diagnosis.get('final_diagnosis', ai_prediction)
    doctor_notes = doctor_diagnosis.get('notes', '')
    next_visit_needed = doctor_diagnosis.get('next_visit_needed', False)
    next_visit_date = doctor_diagnosis.get('next_visit_date', '')
    
    is_smoker = survey_data.get('smoking', False)
    exercise_frequency = survey_data.get('exercise', 0)  # 0: 안함, 1: 가끔, 2: 정기적
    age = survey_data.get('age', 40)
    appointment_date = survey_data.get('appointment_date', '')
    
    recommendations = {
        'priority': 'low',  # low, medium, high, urgent
        'ai_vs_doctor': {
            'ai_diagnosis': ai_prediction,
            'ai_confidence': ai_confidence,
            'doctor_diagnosis': doctor_final_diagnosis,
            'agreement': ai_prediction == doctor_final_diagnosis
        },
        'doctor_notes': doctor_notes,
        'next_visit_info': {
            'needed': next_visit_needed,
            'recommended_date': next_visit_date,
            'scheduled_date': appointment_date
        },
        'smoking_advice': [],
        'exercise_advice': [],
        'medical_advice': [],
        'lifestyle_advice': [],
        'followup_advice': []
    }
    
    # 최종 진단에 따른 우선순위 설정 (의사 진단을 우선)
    final_diagnosis = doctor_final_diagnosis
    
    if final_diagnosis == 'Normal':
        if ai_prediction != 'Normal':
            recommendations['priority'] = 'medium'
            recommendations['medical_advice'].append(
                f"AI는 {ai_prediction}으로 예측했으나, 의사 진단 결과 정상으로 판정되었습니다."
            )
        else:
            recommendations['priority'] = 'low'
            recommendations['medical_advice'].append("AI와 의사 진단 모두 정상으로 판정되었습니다.")
        
        if next_visit_needed:
            recommendations['followup_advice'].append(f"의사 지시에 따라 {next_visit_date}에 재검사가 필요합니다.")
        else:
            recommendations['followup_advice'].append("1년 후 정기 검진을 받으시기 바랍니다.")
            
    elif final_diagnosis == 'Benign':
        recommendations['priority'] = 'medium'
        recommendations['medical_advice'].append("의사 진단 결과 양성 병변으로 확인되었습니다.")
        
        if ai_prediction != 'Benign':
            recommendations['medical_advice'].append(
                f"AI 예측({ai_prediction})과 다른 결과이므로 의사의 판단을 따르시기 바랍니다."
            )
        
        if next_visit_needed:
            recommendations['followup_advice'].append(f"의사 지시에 따라 {next_visit_date}에 추적 검사가 필요합니다.")
        else:
            recommendations['followup_advice'].append("3개월 후 추적 검사를 권장합니다.")
        
    elif final_diagnosis == 'Malignant':
        recommendations['priority'] = 'urgent'
        recommendations['medical_advice'].append("⚠️ 의사 진단 결과 악성 병변으로 확인되었습니다.")
        recommendations['medical_advice'].append("🏥 추가 정밀검사 및 치료 계획이 필요합니다.")
        
        if next_visit_needed:
            recommendations['medical_advice'].append(f"반드시 {next_visit_date}에 내원하시기 바랍니다.")
    
    # 의사 소견이 있는 경우 추가
    if doctor_notes:
        recommendations['medical_advice'].append(f"의사 소견: {doctor_notes}")
    
    # 흡연 관련 권고사항
    if is_smoker:
        if final_diagnosis in ['Benign', 'Malignant']:
            recommendations['smoking_advice'].append("🚭 진단 결과를 고려하여 즉시 금연하시기 바랍니다.")
            recommendations['smoking_advice'].append("금연 클리닉이나 금연 상담 프로그램을 이용하세요.")
        else:
            recommendations['smoking_advice'].append("🚭 폐 건강을 위해 금연을 강력히 권장합니다.")
        
        recommendations['smoking_advice'].append("금연 후에도 정기적인 폐 검진이 필요합니다.")
    else:
        recommendations['smoking_advice'].append("👍 비흡연 습관을 계속 유지하시기 바랍니다.")
    
    # 운동 관련 권고사항
    if exercise_frequency < 2:
        if final_diagnosis in ['Benign', 'Malignant']:
            recommendations['exercise_advice'].append("🏃‍♂️ 치료와 회복을 위해 적절한 운동이 도움됩니다.")
            recommendations['exercise_advice'].append("의사와 상담 후 운동 강도를 조절하세요.")
        else:
            recommendations['exercise_advice'].append("🏃‍♂️ 폐 건강을 위해 규칙적인 운동을 시작하시기 바랍니다.")
        
        recommendations['exercise_advice'].append("주 3회 이상, 30분씩 유산소 운동을 권장합니다.")
    else:
        recommendations['exercise_advice'].append("👍 현재의 규칙적인 운동 습관을 유지하세요.")
    
    # 생활습관 권고사항
    recommendations['lifestyle_advice'].append("🍎 항산화 성분이 풍부한 과일과 채소를 충분히 섭취하세요.")
    recommendations['lifestyle_advice'].append("💧 충분한 수분 섭취를 하시기 바랍니다.")
    recommendations['lifestyle_advice'].append("😴 충분한 수면을 취하시기 바랍니다.")
    
    if final_diagnosis in ['Benign', 'Malignant']:
        recommendations['lifestyle_advice'].append("🏭 미세먼지가 심한 날에는 외출을 자제하고 마스크를 착용하세요.")
        recommendations['lifestyle_advice'].append("🧘‍♀️ 스트레스 관리를 위한 명상이나 요가를 고려해보세요.")
    
    # 예약 정보 관련
    if appointment_date:
        recommendations['followup_advice'].append(f"📅 다음 진료 예약일: {appointment_date}")
    
    # 나이에 따른 추가 권고사항
    if age >= 50:
        recommendations['followup_advice'].append("50세 이상이므로 정기적인 폐암 검진이 중요합니다.")
        if is_smoker or final_diagnosis != 'Normal':
            recommendations['followup_advice'].append("고위험군에 해당하므로 저선량 흉부 CT 검사를 고려하세요.")
    
    return recommendations


def lung_preprocessing(image_path):
    """
    CT 이미지 폐 전처리 함수

    Args:
        image_path: 이미지 경로 (문자열) 또는 이미지 배열

    Returns:
        lung_mask: 폐 영역 마스크 (0-1)
        lung_image: 전처리된 폐 이미지 (0-1)
        original_image: 원본 이미지 (0-1)
    """

    # 이미지 로드
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, None
    else:
        img = image_path

    original_normalized = img.astype(np.float32) / 255.0
    img_height, img_width = original_normalized.shape
    center_x, center_y = img_width // 2, img_height // 2

    # 1. 몸통 추출
    img_hu_approx = (original_normalized * 2000) - 1000
    body_mask = img_hu_approx > -100

    kernel = np.ones((5, 5), np.uint8)
    body_mask = cv2.morphologyEx(body_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 몸통 선택 (중앙에 가장 가까운 큰 영역)
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        min_body_area = body_mask.size * 0.15
        max_body_area = body_mask.size * 0.8

        best_contour = None
        min_distance = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_body_area <= area <= max_body_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)

                    if distance < min_distance:
                        min_distance = distance
                        best_contour = contour

        if best_contour is not None:
            body_mask = np.zeros_like(body_mask)
            cv2.fillPoly(body_mask, [best_contour], 1)
        else:
            return None, None, None
    else:
        return None, None, None

    # 2. 베드 영역 제거
    bed_mask = np.zeros_like(body_mask, dtype=bool)

    # 하단 10%, 좌우 8% 제거
    bottom_region = int(img_height * 0.10)
    side_margin = int(img_width * 0.08)
    bed_mask[-bottom_region:, :] = True
    bed_mask[:, :side_margin] = True
    bed_mask[:, -side_margin:] = True

    # 몸통 외부 어두운 영역 제거
    expanded_body = cv2.dilate(body_mask, np.ones((15, 15), np.uint8), iterations=2)
    outside_body = ~expanded_body.astype(bool)
    dark_areas = original_normalized < 0.1
    bed_mask = bed_mask | (outside_body & dark_areas)

    # 3. 하얀 조직 및 혈관 제거
    enhanced = np.clip(original_normalized * 1.2 + 0.1, 0, 1)
    white_mask = (enhanced * body_mask) > 0.75

    # 🆕 혈관/염증 영역 제거 (너무 밝은 부분)
    vessel_mask = (original_normalized > 0.7) & body_mask.astype(bool)

    # 폐 영역 마스크 생성 (더 엄격한 밝기 범위)
    intensity_mask = (original_normalized >= 0.18) & (original_normalized <= 0.55)
    lung_mask = body_mask.astype(bool) & ~white_mask & ~bed_mask & ~vessel_mask & intensity_mask

    # 4. 몸통 경계 수축
    eroded_body = cv2.erode(body_mask, np.ones((7, 7), np.uint8), iterations=2)
    lung_mask = lung_mask & eroded_body.astype(bool)

    # 5. 폐 영역 선택 (화면 3등분 기준으로 좌우선에서 가까운 순서로 선택)
    contours, _ = cv2.findContours(lung_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    body_area = np.sum(body_mask)
    min_area = body_area * 0.005
    max_area = body_area * 0.35

    # 화면을 세등분: 좌측선(1/3), 우측선(2/3)
    left_line = img_width / 3
    right_line = img_width * 2 / 3

    # 유효한 폐 영역 찾기
    valid_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 위치 검증
                if (cy < img_height * 0.7 and
                    img_width * 0.15 < cx < img_width * 0.85):

                    # 좌측선과 우측선으로부터의 거리 계산
                    dist_to_left_line = abs(cx - left_line)
                    dist_to_right_line = abs(cx - right_line)
                    min_dist_to_lines = min(dist_to_left_line, dist_to_right_line)

                    # 좌측/우측 구분
                    is_left_region = cx < center_x

                    valid_regions.append((contour, area, cx, cy, min_dist_to_lines, is_left_region))

    # 좌우선에 가까운 순서로 정렬
    valid_regions.sort(key=lambda x: x[4])  # min_dist_to_lines 기준으로 정렬

    # 좌우 폐 선택
    selected = []
    left_selected = False
    right_selected = False

    for region in valid_regions:
        contour, area, cx, cy, dist_to_lines, is_left = region

        # 좌측 영역 선택
        if is_left and not left_selected:
            selected.append(region)
            left_selected = True
        # 우측 영역 선택
        elif not is_left and not right_selected:
            selected.append(region)
            right_selected = True

        # 좌우 모두 선택되면 종료
        if left_selected and right_selected:
            break

    # 좌우 중 하나만 선택된 경우, 추가로 하나 더 선택 시도
    if len(selected) == 1:
        selected_side = selected[0][5]  # 이미 선택된 쪽이 좌측인지 우측인지
        selected_cx = selected[0][2]
        selected_area = selected[0][1]

        # 반대편에서 가장 가까운 영역 찾기
        for i, region in enumerate(valid_regions):
            # 좌표와 면적으로 중복 확인
            is_already_selected = any(selected_cx == s[2] and selected_area == s[1] for s in selected)

            if not is_already_selected and region[5] != selected_side:
                # 면적 비율 검증
                area_ratio = min(selected_area, region[1]) / max(selected_area, region[1])
                if area_ratio > 0.15:  # 면적 비율이 합리적인 경우만
                    selected.append(region)
                    break

    # 아직도 하나만 선택된 경우, 같은 쪽에서라도 추가 선택
    if len(selected) == 1:
        selected_cx = selected[0][2]
        selected_area = selected[0][1]

        for i, region in enumerate(valid_regions):
            # 좌표와 면적으로 중복 확인
            is_already_selected = any(selected_cx == s[2] and selected_area == s[1] for s in selected)

            if not is_already_selected:
                # 거리 검증 (너무 가깝지 않은 경우만)
                cx_diff = abs(selected_cx - region[2])
                if cx_diff > img_width * 0.1:
                    area_ratio = min(selected_area, region[1]) / max(selected_area, region[1])
                    if area_ratio > 0.15:
                        selected.append(region)
                        break

    # 선택된 영역이 없는 경우, 면적이 가장 큰 두 영역 선택
    if len(selected) == 0 and len(valid_regions) > 0:
        # 면적 순으로 정렬
        valid_regions.sort(key=lambda x: x[1], reverse=True)
        selected = valid_regions[:min(2, len(valid_regions))]

    # 선택된 영역 마스크에 추가
    for region in selected:
        cv2.fillPoly(final_mask, [region[0]], 1)

    # 추가: 세밀한 폐 내부 채우기 기능
    if selected:  # 선택된 영역이 있을 때만 채우기 수행

        # 1. 작은 구멍만 채우기 (보수적 접근)
        small_fill_kernel = np.ones((3, 3), np.uint8)
        filled_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, small_fill_kernel, iterations=2)

        # 2. 선택적 내부 채우기 (컨벡스 헐 대신 제한적 확장)
        def selective_fill(mask, original_img, body_mask):
            """폐 조직 영역만 선별적으로 채우기"""
            result = mask.copy()

            # 폐 조직 같은 밝기 영역 찾기 (기존 폐 마스크 주변)
            lung_intensity_min = 0.15
            lung_intensity_max = 0.6

            # 기존 폐 영역 확장
            expand_kernel = np.ones((3, 3), np.uint8)
            expanded = cv2.dilate(mask, expand_kernel, iterations=2)

            # 확장된 영역 중에서 폐 조직 밝기에 해당하는 부분만 선택
            lung_like_areas = (original_img >= lung_intensity_min) & (original_img <= lung_intensity_max)
            valid_expansion = expanded & body_mask.astype(np.uint8) & lung_like_areas.astype(np.uint8)

            # 뼈/혈관 영역 제외 (더 엄격하게)
            exclude_bright = original_img > 0.7  # 밝은 혈관/뼈
            exclude_very_dark = original_img < 0.12  # 너무 어두운 공기

            valid_expansion = valid_expansion & ~exclude_bright.astype(np.uint8) & ~exclude_very_dark.astype(np.uint8)

            return valid_expansion

        # 선별적 채우기 적용
        filled_mask = selective_fill(filled_mask, original_normalized, body_mask)

        # 3. 형태학적 정리 (부드럽게)
        smooth_kernel = np.ones((2, 2), np.uint8)
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)

        # 4. 원본 영역과 결합 (기존 + 새로 채운 부분)
        # 기존 영역은 보존하고 새로 채운 부분만 추가
        final_filled = final_mask | filled_mask

        # 5. 최종 안전 체크 - 너무 커지지 않도록 제한
        contours_check, _ = cv2.findContours(final_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        checked_mask = np.zeros_like(final_filled)

        for contour in contours_check:
            area = cv2.contourArea(contour)
            # 면적이 너무 크지 않은 경우만 허용 (몸통의 40% 이하)
            if min_area <= area <= body_area * 0.4:
                cv2.fillPoly(checked_mask, [contour], 1)

        # 최종 마스크 적용
        final_mask = checked_mask

    # 최종 결과
    lung_mask_final = final_mask.astype(bool)
    lung_image = original_normalized * lung_mask_final

    return lung_mask_final, lung_image, original_normalized

def preprocess_image_with_improved_segmentation(image_path):
    """
    개선된 이미지 전처리 함수 (Flask 인터페이스와 호환)
    """
    
    try:
        # 개선된 폐 전처리 적용
        lung_mask, lung_img, original_img = lung_preprocessing(image_path)
        
        if lung_mask is None:
            return None, None
        
        # 폐 비율 검증
        lung_ratio = np.sum(lung_mask) / lung_mask.size
        if not (0.003 < lung_ratio < 0.45):
            print(f"폐 비율 검증 실패: {lung_ratio:.3f}")
            return None, None
        
        # 512x512로 리사이즈
        processed_img = cv2.resize(lung_img, (512, 512))
        original_resized = cv2.resize(original_img, (512, 512))
        
        # 정규화 (0-1 범위)
        processed_img = processed_img.astype(np.float32)
        original_resized = original_resized.astype(np.float32)
        
        return processed_img, original_resized
        
    except Exception as e:
        print(f"전처리 오류: {e}")
        return None, None

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    img_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)

    x = MaxPooling2D(2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D(2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D(2)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D(2)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D()(x)

    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=img_input, outputs=output)
    return model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Grad-CAM 히트맵 생성"""
    preds = model.predict(img_array)
    pred_index = tf.argmax(preds[0])
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_gradcam_image(original_img, heatmap, alpha=0.6):
    """Grad-CAM 오버레이 이미지 생성 (원본 이미지에 히트맵 오버레이)"""
    # 히트맵을 원본 이미지 크기에 맞게 리사이즈
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    
    # 히트맵을 컬러맵으로 변환 (jet 컬러맵 사용)
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]
    
    # 원본 이미지를 3채널로 변환 (그레이스케일을 RGB로)
    if len(original_img.shape) == 2:
        original_img_3ch = np.stack([original_img]*3, axis=-1)
    else:
        original_img_3ch = original_img
    
    # 원본 이미지 정규화 (0-1 범위로)
    original_img_normalized = original_img_3ch / np.max(original_img_3ch) if np.max(original_img_3ch) > 0 else original_img_3ch
    
    # 히트맵에서 투명도 마스크 생성 (히트맵 값이 낮은 곳은 투명하게)
    transparency_mask = heatmap_resized
    transparency_mask = np.stack([transparency_mask]*3, axis=-1)
    
    # 알파 블렌딩: 히트맵 강도에 따라 투명도 조절
    # 히트맵이 강한 곳은 더 불투명하게, 약한 곳은 더 투명하게
    dynamic_alpha = alpha * transparency_mask
    
    # 최종 이미지 합성
    superimposed_img = (heatmap_colored * dynamic_alpha + 
                       original_img_normalized * (1 - dynamic_alpha))
    
    # 0-255 범위로 변환
    superimposed_img = np.uint8(255 * superimposed_img)
    
    return superimposed_img


def make_multi_layer_gradcam_heatmap(img_array, model, layer_names, weights=None):
    """
    여러 레이어의 Grad-CAM 히트맵을 생성하고 합치기
    
    Args:
        img_array: 입력 이미지 배열
        model: 학습된 모델
        layer_names: 히트맵을 생성할 레이어 이름들의 리스트 (최상층부터 순서대로)
        weights: 각 레이어의 가중치 (None이면 균등 가중치)
    
    Returns:
        combined_heatmap: 합쳐진 히트맵
        individual_heatmaps: 각 레이어별 개별 히트맵들
    """
    if weights is None:
        weights = [1.0] * len(layer_names)
    
    # 가중치 정규화
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    preds = model.predict(img_array)
    pred_index = tf.argmax(preds[0])
    
    individual_heatmaps = []
    
    for layer_name in layer_names:
        # 각 레이어에 대해 Grad-CAM 모델 생성
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, pred_index]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)

        # 정규화
        if tf.math.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.math.reduce_max(heatmap)
        
        individual_heatmaps.append(heatmap.numpy())
    
    # 모든 히트맵을 동일한 크기로 리사이즈
    target_size = individual_heatmaps[0].shape
    resized_heatmaps = []
    
    for heatmap in individual_heatmaps:
        if heatmap.shape != target_size:
            # OpenCV를 사용해서 리사이즈
            resized = cv2.resize(heatmap, (target_size[1], target_size[0]))
            resized_heatmaps.append(resized)
        else:
            resized_heatmaps.append(heatmap)
    
    # 가중 평균으로 히트맵 합치기
    combined_heatmap = np.zeros_like(resized_heatmaps[0])
    for i, heatmap in enumerate(resized_heatmaps):
        combined_heatmap += weights[i] * heatmap
    
    return combined_heatmap, individual_heatmaps

def get_conv_layer_names(model, top_n=3):
    """
    모델에서 상위 n개의 합성곱 레이어 이름을 가져오기
    
    Args:
        model: 케라스 모델
        top_n: 가져올 레이어 수 (최상위부터)
    
    Returns:
        layer_names: 레이어 이름들의 리스트
    """
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
    
    # 최상위 레이어부터 반환 (역순으로)
    return conv_layers[-top_n:]





# 모델 로드
print("🔄 모델 로딩 중...")
try:
    # 먼저 모델 구조 생성
    model = build_model()

    model.summary()
    # 웨이트 로드
    model.load_weights(h5_path, by_name=True)
    
    
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """이미지 예측 및 Grad-CAM 생성"""
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'})
    
    if file and allowed_file(file.filename):
        try:
            # 파일 저장
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 개선된 이미지 전처리 (원본 이미지도 함께 반환)
            processed_img, original_img = preprocess_image_with_improved_segmentation(filepath)
            if processed_img is None:
                # 전처리 실패 시 관리자에게 알림 메시지
                return jsonify({
                    'error': '전처리 실패',
                    'message': '이 CT 이미지는 자동 분석이 어려워 관리자에게 전송되었습니다. 수동 검토 후 결과를 알려드리겠습니다.',
                    'admin_notification': True
                })
            
            # 모델 입력 형태로 변환
            img_array = processed_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            meta_array = np.array([[0.1, 0.1]])  # 더미 메타데이터
            
            # 예측
            predictions = model.predict(img_array)
            pred_class = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_class])

            # 모델의 상위 3개 합성곱 레이어 이름 자동 추출
            top_conv_layers = get_conv_layer_names(model, top_n=6)
            print("Top conv layers:", top_conv_layers)
            
            # 각 레이어에 다른 가중치 부여 (최상위 레이어에 더 높은 가중치)
            layer_weights = [0.2, 0.2, 0.2,0.2,0.2,0.2]  # 최상위, 중간, 하위 순서
            
            # 다중 레이어 Grad-CAM 시각화
            combined_heatmap, individual_heatmaps = make_multi_layer_gradcam_heatmap(img_array, model, top_conv_layers, weights=layer_weights)
       
            gradcam_img = create_gradcam_image(original_img, individual_heatmaps[-1])
                
            # 이미지를 base64로 인코딩
            _, buffer = cv2.imencode('.png', gradcam_img)
            gradcam_b64 = base64.b64encode(buffer).decode('utf-8')


            
            # 원본 이미지도 base64로 인코딩
            _, orig_buffer = cv2.imencode('.png', original_img * 255)
            original_b64 = base64.b64encode(orig_buffer).decode('utf-8')
            
            # 전처리된 이미지도 base64로 인코딩
            _, proc_buffer = cv2.imencode('.png', processed_img * 255)
            processed_b64 = base64.b64encode(proc_buffer).decode('utf-8')
            
            # 결과 반환
            result = {
                'prediction': CLASS_NAMES[pred_class],
                'confidence': round(confidence * 100, 2),
                'color': CLASS_COLORS[pred_class],
                'all_predictions': {
                    CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2) 
                    for i in range(len(CLASS_NAMES))
                },
                'original_image': f"data:image/png;base64,{original_b64}",
                'processed_image': f"data:image/png;base64,{processed_b64}",
                'gradcam_image': f"data:image/png;base64,{gradcam_b64}" if gradcam_b64 else None,
                'preprocessing_success': True,
                'need_doctor_input': True  # 의사 진단 입력 필요 플래그
            }
            
            # 임시 파일 삭제
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            # 임시 파일 삭제 (실패 시에도)
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'error': '처리 중 오류 발생',
                'message': f'시스템 오류가 발생했습니다: {str(e)}',
                'admin_notification': True
            })
    
    return jsonify({'error': '잘못된 파일 형식입니다.'})

@app.route('/doctor_diagnosis', methods=['POST'])
def doctor_diagnosis():
    """의사 진단 입력 처리"""
    try:
        data = request.get_json()
        
        # AI 예측 결과
        ai_result = data.get('ai_result', {})
        
        # 의사 진단 정보
        doctor_data = {
            'final_diagnosis': data.get('final_diagnosis'),
            'notes': data.get('notes', ''),
            'next_visit_needed': data.get('next_visit_needed', False),
            'next_visit_date': data.get('next_visit_date', '')
        }
        
        return jsonify({
            'success': True,
            'message': '의사 진단이 저장되었습니다.',
            'need_survey': True,
            'doctor_diagnosis': doctor_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'의사 진단 저장 중 오류 발생: {str(e)}'
        })

@app.route('/health_recommendations', methods=['POST'])
def health_recommendations():
    """설문조사 결과를 받아 개인화된 건강 권고사항 및 예약 완료 처리"""
    try:
        data = request.get_json()
        
        # AI 예측 결과
        prediction_result = data.get('prediction_result', {})
        
        # 의사 진단 결과
        doctor_diagnosis = data.get('doctor_diagnosis', {})
        
        # 설문조사 결과
        survey_data = {
            'smoking': data.get('smoking', False),
            'exercise': int(data.get('exercise', 0)),
            'age': int(data.get('age', 40)),
            'appointment_date': data.get('appointment_date', '')
        }
        
        # 건강 권고사항 생성
        recommendations = generate_health_recommendations(prediction_result, doctor_diagnosis, survey_data)
        
        # 예약 완료 처리
        appointment_info = None
        if survey_data['appointment_date']:
            appointment_info = {
                'date': survey_data['appointment_date'],
                'status': 'confirmed',
                'message': f"{survey_data['appointment_date']}에 진료 예약이 완료되었습니다."
            }
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'appointment': appointment_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'권고사항 생성 중 오류 발생: {str(e)}'
        })

# HTML 템플릿 (의사 진단 입력 및 예약 기능 추가)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FocusNet-LC 폐암 진단</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            margin: 20px 0;
        }
        
        .file-input {
            display: none;
        }
        
        .file-input-button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .file-input-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .predict-button {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 1.2em;
            cursor: pointer;
            margin-left: 20px;
            transition: all 0.3s;
        }
        
        .predict-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .predict-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .results-section {
            display: none;
            margin-top: 40px;
        }
        
        .prediction-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .prediction-result {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .confidence {
            font-size: 1.5em;
            margin-bottom: 20px;
        }
        
        .all-predictions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .prediction-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .images-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        
        .image-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        
        .image-card h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .image-card img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .doctor-section {
            display: none;
            background: #fff9e6;
            border: 2px solid #ffc107;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
        }
        
        .doctor-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            text-align: center;
            color: #856404;
            font-weight: bold;
        }
        
        .doctor-form {
            max-width: 700px;
            margin: 0 auto;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        .form-group label {
            display: block;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        .form-group select, .form-group textarea, .form-group input[type="date"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            font-family: inherit;
        }
        
        .form-group textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        .form-group input[type="radio"] {
            margin-right: 10px;
        }
        
        .radio-group {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }
        
        .radio-option {
            display: flex;
            align-items: center;
            padding: 10px 15px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .radio-option:hover {
            background: #f8f9fa;
            border-color: #007bff;
        }
        
        .radio-option input[type="radio"]:checked + span {
            color: #007bff;
            font-weight: bold;
        }
        
        .submit-doctor-button {
            background: linear-gradient(135deg, #ffc107, #ff8f00);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
        }
        
        .submit-doctor-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .survey-section {
            display: none;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
        }
        
        .survey-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            text-align: center;
            color: #333;
        }
        
        .survey-form {
            max-width: 600px;
            margin: 0 auto;
        }
        
        .survey-form .form-group input[type="radio"] {
            margin-right: 10px;
        }
        
        .survey-form .radio-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .survey-form .radio-option {
            display: flex;
            align-items: center;
            padding: 10px;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid #ddd;
        }
        
        .survey-form .radio-option:hover {
            background: #e9ecef;
        }
        
        .survey-form input[type="number"], .survey-form input[type="date"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
        }
        
        .submit-survey-button {
            background: linear-gradient(135deg, #17a2b8, #138496);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
        }
        
        .submit-survey-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .recommendations-section {
            display: none;
            margin-top: 30px;
        }
        
        .recommendations-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
        }
        
        .priority-banner {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .priority-low {
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        
        .priority-medium {
            background: #fff3cd;
            color: #856404;
            border: 2px solid #ffeaa7;
        }
        
        .priority-high {
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        
        .priority-urgent {
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .ai-vs-doctor {
            background: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }
        
        .diagnosis-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        
        .diagnosis-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .agreement-status {
            text-align: center;
            margin-top: 15px;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
        }
        
        .agreement-true {
            background: #d4edda;
            color: #155724;
        }
        
        .agreement-false {
            background: #f8d7da;
            color: #721c24;
        }
        
        .appointment-card {
            background: #e8f5e8;
            border: 2px solid #28a745;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
        }
        
        .appointment-confirmed {
            color: #155724;
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .recommendation-category {
            margin-bottom: 25px;
        }
        
        .recommendation-category h4 {
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ddd;
        }
        
        .recommendation-list {
            list-style: none;
            padding: 0;
        }
        
        .recommendation-list li {
            background: white;
            padding: 12px 15px;
            margin-bottom: 8px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .recommendation-list li:before {
            content: "✓ ";
            color: #28a745;
            font-weight: bold;
            margin-right: 8px;
        }
        
        .smoking-advice li {
            border-left-color: #dc3545;
        }
        
        .exercise-advice li {
            border-left-color: #28a745;
        }
        
        .medical-advice li {
            border-left-color: #ffc107;
        }
        
        .lifestyle-advice li {
            border-left-color: #17a2b8;
        }
        
        .followup-advice li {
            border-left-color: #6f42c1;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            background: #dc3545;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        .admin-message {
            background: #ffc107;
            color: #856404;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
            text-align: center;
        }
        
        .admin-message h4 {
            margin-bottom: 10px;
            color: #856404;
        }
        
        @media (max-width: 768px) {
            .images-container {
                grid-template-columns: 1fr;
            }
            
            .predict-button {
                margin-left: 0;
                margin-top: 10px;
            }
            
            .radio-group {
                flex-direction: column;
            }
            
            .diagnosis-comparison {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Bread Scan</h1>
            <p>AI 기반 폐암 진단 시스템 (서강대 AISW 텐서플로 활용기초)</p>
        </div>
        
        <div class="content">
            <div class="upload-section">
                <h2>CT 이미지를 업로드하세요</h2>
                <div class="file-input-wrapper">
                    <input type="file" id="fileInput" class="file-input" accept=".png,.jpg,.jpeg">
                    <button class="file-input-button" onclick="document.getElementById('fileInput').click()">
                        📁 파일 선택
                    </button>
                </div>
                <button class="predict-button" id="predictButton" disabled onclick="predict()">
                    🔍 진단 시작
                </button>
                <div id="fileName" style="margin-top: 10px; color: #666;"></div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>AI가 이미지를 분석하고 있습니다...</p>
            </div>
            
            <div class="error-message" id="errorMessage"></div>
            
            <div class="admin-message" id="adminMessage">
                <h4>⚠️ 전처리 실패</h4>
                <p id="adminMessageText"></p>
            </div>
            
            <div class="results-section" id="resultsSection">
                <div class="prediction-card">
                    <div class="prediction-result" id="predictionResult"></div>
                    <div class="confidence" id="confidence"></div>
                    <div class="all-predictions" id="allPredictions"></div>
                </div>
                
                <div class="images-container">
                    <div class="image-card">
                        <h3>📷 원본 이미지</h3>
                        <img id="originalImage" src="" alt="원본 이미지">
                    </div>
                    <div class="image-card">
                        <h3>🔍 전처리된 이미지</h3>
                        <img id="processedImage" src="" alt="전처리된 이미지">
                    </div>
                    <div class="image-card">
                        <h3>🎯 Grad-CAM 히트맵</h3>
                        <img id="gradcamImage" src="" alt="Grad-CAM 히트맵">
                    </div>
                </div>
            </div>
            
            <!-- 의사 진단 섹션 -->
            <div class="doctor-section" id="doctorSection">
                <h3 class="doctor-title">🩺 의사 진단 및 소견</h3>
                <p style="text-align: center; margin-bottom: 30px; color: #856404;">
                    AI 분석 결과를 검토하시고 최종 진단을 입력해주세요.
                </p>
                
                <form class="doctor-form" id="doctorForm">
                    <div class="form-group">
                        <label>최종 진단 결과</label>
                        <select name="final_diagnosis" required>
                            <option value="">선택해주세요</option>
                            <option value="Normal">정상 (Normal)</option>
                            <option value="Benign">양성 병변 (Benign)</option>
                            <option value="Malignant">악성 병변 (Malignant)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>의사 소견 및 추가 의견</label>
                        <textarea name="notes" placeholder="진단 소견, 추가 검사 필요성, 주의사항 등을 입력해주세요..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label>다음 진료 필요 여부</label>
                        <div class="radio-group">
                            <label class="radio-option">
                                <input type="radio" name="next_visit_needed" value="false" required>
                                <span>불필요 (정기 검진으로 충분)</span>
                            </label>
                            <label class="radio-option">
                                <input type="radio" name="next_visit_needed" value="true" required>
                                <span>필요 (추적 관찰 또는 정밀검사)</span>
                            </label>
                        </div>
                    </div>
                    
                    <div class="form-group" id="nextVisitDateGroup" style="display: none;">
                        <label>권장 다음 진료일</label>
                        <input type="date" name="next_visit_date" min="">
                    </div>
                    
                    <button type="submit" class="submit-doctor-button">
                        💾 진단 완료
                    </button>
                </form>
            </div>
            
            <!-- 설문조사 섹션 -->
            <div class="survey-section" id="surveySection">
                <h3 class="survey-title">📋 건강 설문조사 및 예약</h3>
                <p style="text-align: center; margin-bottom: 30px; color: #666;">
                    더 정확한 건강 권고사항을 위해 간단한 설문에 참여해주세요.
                </p>
                
                <form class="survey-form" id="surveyForm">
                    <div class="form-group">
                        <label>1. 현재 흡연 상태는 어떠신가요?</label>
                        <div class="radio-group">
                            <label class="radio-option">
                                <input type="radio" name="smoking" value="false" required>
                                <span>🚭 비흡연자 (흡연한 적 없음)</span>
                            </label>
                            <label class="radio-option">
                                <input type="radio" name="smoking" value="false" required>
                                <span>🚫 금연자 (과거 흡연, 현재 금연)</span>
                            </label>
                            <label class="radio-option">
                                <input type="radio" name="smoking" value="true" required>
                                <span>🚬 현재 흡연자</span>
                            </label>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>2. 평소 운동은 얼마나 자주 하시나요?</label>
                        <div class="radio-group">
                            <label class="radio-option">
                                <input type="radio" name="exercise" value="0" required>
                                <span>😴 운동을 거의 하지 않음</span>
                            </label>
                            <label class="radio-option">
                                <input type="radio" name="exercise" value="1" required>
                                <span>🚶‍♂️ 가끔 운동함 (주 1-2회)</span>
                            </label>
                            <label class="radio-option">
                                <input type="radio" name="exercise" value="2" required>
                                <span>🏃‍♂️ 정기적으로 운동함 (주 3회 이상)</span>
                            </label>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="age">3. 연령을 입력해주세요:</label>
                        <input type="number" id="age" name="age" min="1" max="120" value="40" required>
                    </div>
                    
                    <div class="form-group" id="appointmentGroup" style="display: none;">
                        <label for="appointment_date">4. 다음 진료 예약일을 선택해주세요:</label>
                        <input type="date" id="appointment_date" name="appointment_date" min="">
                    </div>
                    
                    <button type="submit" class="submit-survey-button">
                        📊 건강 권고사항 받기
                    </button>
                </form>
            </div>
            
            <!-- 건강 권고사항 섹션 -->
            <div class="recommendations-section" id="recommendationsSection">
                <div class="recommendations-card">
                    <h3 style="text-align: center; margin-bottom: 20px; color: #333;">
                        💡 개인화된 건강 권고사항
                    </h3>
                    
                    <div id="priorityBanner" class="priority-banner"></div>
                    
                    <!-- AI vs 의사 진단 비교 -->
                    <div id="aiVsDoctorComparison" class="ai-vs-doctor">
                        <h4 style="margin-bottom: 15px; color: #1976d2;">🤖 AI vs 👨‍⚕️ 의사 진단 비교</h4>
                        <div class="diagnosis-comparison">
                            <div class="diagnosis-item">
                                <h5>AI 진단</h5>
                                <div id="aiDiagnosisResult"></div>
                                <small id="aiConfidence"></small>
                            </div>
                            <div class="diagnosis-item">
                                <h5>의사 진단</h5>
                                <div id="doctorDiagnosisResult"></div>
                            </div>
                        </div>
                        <div id="agreementStatus" class="agreement-status"></div>
                        <div id="doctorNotesDisplay" style="margin-top: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px;"></div>
                    </div>
                    
                    <!-- 예약 완료 카드 -->
                    <div id="appointmentCard" class="appointment-card" style="display: none;">
                        <div class="appointment-confirmed">✅ 예약 완료</div>
                        <div id="appointmentDetails"></div>
                    </div>
                    
                    <div id="medicalAdvice" class="recommendation-category">
                        <h4>🏥 의료 관련 권고사항</h4>
                        <ul class="recommendation-list medical-advice"></ul>
                    </div>
                    
                    <div id="smokingAdvice" class="recommendation-category">
                        <h4>🚭 흡연 관련 권고사항</h4>
                        <ul class="recommendation-list smoking-advice"></ul>
                    </div>
                    
                    <div id="exerciseAdvice" class="recommendation-category">
                        <h4>🏃‍♂️ 운동 관련 권고사항</h4>
                        <ul class="recommendation-list exercise-advice"></ul>
                    </div>
                    
                    <div id="lifestyleAdvice" class="recommendation-category">
                        <h4>🌱 생활습관 권고사항</h4>
                        <ul class="recommendation-list lifestyle-advice"></ul>
                    </div>
                    
                    <div id="followupAdvice" class="recommendation-category">
                        <h4>📅 추후 관리 계획</h4>
                        <ul class="recommendation-list followup-advice"></ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentPredictionResult = null;
        let currentDoctorDiagnosis = null;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            const predictButton = document.getElementById('predictButton');
            const fileName = document.getElementById('fileName');
            
            if (file) {
                fileName.textContent = `선택된 파일: ${file.name}`;
                predictButton.disabled = false;
            } else {
                fileName.textContent = '';
                predictButton.disabled = true;
            }
        });
        
        // 의사 진단 폼 이벤트 리스너
        document.getElementById('doctorForm').addEventListener('submit', function(e) {
            e.preventDefault();
            submitDoctorDiagnosis();
        });
        
        // 다음 진료 필요 여부에 따른 날짜 필드 표시/숨김
        document.querySelectorAll('input[name="next_visit_needed"]').forEach(radio => {
            radio.addEventListener('change', function() {
                const dateGroup = document.getElementById('nextVisitDateGroup');
                const dateInput = document.querySelector('input[name="next_visit_date"]');
                
                if (this.value === 'true') {
                    dateGroup.style.display = 'block';
                    dateInput.required = true;
                    // 최소 날짜를 내일로 설정
                    const tomorrow = new Date();
                    tomorrow.setDate(tomorrow.getDate() + 1);
                    dateInput.min = tomorrow.toISOString().split('T')[0];
                } else {
                    dateGroup.style.display = 'none';
                    dateInput.required = false;
                    dateInput.value = '';
                }
            });
        });
        
        document.getElementById('surveyForm').addEventListener('submit', function(e) {
            e.preventDefault();
            submitSurvey();
        });
        
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 10000);
        }
        
        function showAdminMessage(message) {
            const adminDiv = document.getElementById('adminMessage');
            const adminText = document.getElementById('adminMessageText');
            adminText.textContent = message;
            adminDiv.style.display = 'block';
            
            setTimeout(() => {
                adminDiv.style.display = 'none';
            }, 15000);
        }
        
        async function predict() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('파일을 선택해주세요.');
                return;
            }
            
            // UI 상태 변경
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('doctorSection').style.display = 'none';
            document.getElementById('surveySection').style.display = 'none';
            document.getElementById('recommendationsSection').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('adminMessage').style.display = 'none';
            document.getElementById('predictButton').disabled = true;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.error) {
                    if (result.admin_notification) {
                        showAdminMessage(result.message || result.error);
                    } else {
                        showError(result.error);
                    }
                    return;
                }
                
                // 현재 예측 결과 저장
                currentPredictionResult = result;
                
                // 결과 표시
                document.getElementById('predictionResult').textContent = result.prediction;
                document.getElementById('predictionResult').style.color = result.color;
                document.getElementById('confidence').textContent = `신뢰도: ${result.confidence}%`;
                
                // 모든 예측 결과 표시
                const allPredDiv = document.getElementById('allPredictions');
                allPredDiv.innerHTML = '';
                for (const [className, confidence] of Object.entries(result.all_predictions)) {
                    const div = document.createElement('div');
                    div.className = 'prediction-item';
                    div.innerHTML = `<strong>${className}</strong><br>${confidence}%`;
                    allPredDiv.appendChild(div);
                }
                
                // 이미지 표시
                document.getElementById('originalImage').src = result.original_image;
                document.getElementById('processedImage').src = result.processed_image;
                if (result.gradcam_image) {
                    document.getElementById('gradcamImage').src = result.gradcam_image;
                }
                
                document.getElementById('resultsSection').style.display = 'block';
                
                // 의사 진단 섹션 표시
                if (result.need_doctor_input) {
                    setTimeout(() => {
                        document.getElementById('doctorSection').style.display = 'block';
                        document.getElementById('doctorSection').scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'start' 
                        });
                    }, 1000);
                }
                
            } catch (error) {
                showError('서버 연결 오류가 발생했습니다.');
                console.error('Error:', error);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictButton').disabled = false;
            }
        }
        
        async function submitDoctorDiagnosis() {
            if (!currentPredictionResult) {
                showError('진단 결과가 없습니다. 먼저 이미지 진단을 수행해주세요.');
                return;
            }
            
            const formData = new FormData(document.getElementById('doctorForm'));
            const doctorData = {
                ai_result: currentPredictionResult,
                final_diagnosis: formData.get('final_diagnosis'),
                notes: formData.get('notes'),
                next_visit_needed: formData.get('next_visit_needed') === 'true',
                next_visit_date: formData.get('next_visit_date')
            };
            
            try {
                const response = await fetch('/doctor_diagnosis', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(doctorData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentDoctorDiagnosis = result.doctor_diagnosis;
                    
                    // 설문조사 섹션 표시
                    setTimeout(() => {
                        document.getElementById('surveySection').style.display = 'block';
                        
                        // 다음 진료가 필요한 경우에만 예약 필드 표시
                        const appointmentGroup = document.getElementById('appointmentGroup');
                        const appointmentInput = document.getElementById('appointment_date');
                        
                        if (result.doctor_diagnosis.next_visit_needed) {
                            appointmentGroup.style.display = 'block';
                            appointmentInput.required = true;
                            
                            // 권장 날짜가 있으면 기본값으로 설정
                            if (result.doctor_diagnosis.next_visit_date) {
                                appointmentInput.value = result.doctor_diagnosis.next_visit_date;
                            }
                            
                            // 최소 날짜를 내일로 설정
                            const tomorrow = new Date();
                            tomorrow.setDate(tomorrow.getDate() + 1);
                            appointmentInput.min = tomorrow.toISOString().split('T')[0];
                        } else {
                            // 다음 진료가 필요하지 않은 경우 예약 필드 숨김
                            appointmentGroup.style.display = 'none';
                            appointmentInput.required = false;
                            appointmentInput.value = '';
                        }
                        
                        document.getElementById('surveySection').scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'start' 
                        });
                    }, 500);
                } else {
                    showError(result.error || '의사 진단 저장 중 오류가 발생했습니다.');
                }
                
            } catch (error) {
                showError('서버 연결 오류가 발생했습니다.');
                console.error('Error:', error);
            }
        }
        
        async function submitSurvey() {
            if (!currentPredictionResult || !currentDoctorDiagnosis) {
                showError('진단 결과가 없습니다. 처음부터 다시 진행해주세요.');
                return;
            }
            
            const formData = new FormData(document.getElementById('surveyForm'));
            const surveyData = {
                prediction_result: currentPredictionResult,
                doctor_diagnosis: currentDoctorDiagnosis,
                smoking: formData.get('smoking') === 'true',
                exercise: parseInt(formData.get('exercise')),
                age: parseInt(formData.get('age')),
                appointment_date: formData.get('appointment_date')
            };
            
            try {
                const response = await fetch('/health_recommendations', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(surveyData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayRecommendations(result.recommendations, result.appointment);
                } else {
                    showError(result.error || '권고사항 생성 중 오류가 발생했습니다.');
                }
                
            } catch (error) {
                showError('서버 연결 오류가 발생했습니다.');
                console.error('Error:', error);
            }
        }
        
        function displayRecommendations(recommendations, appointment) {
            // 우선순위 배너 설정
            const priorityBanner = document.getElementById('priorityBanner');
            const priorityTexts = {
                'low': '✅ 저위험군 - 예방 관리 중심',
                'medium': '⚠️ 중위험군 - 적극적 관리 필요',
                'high': '🚨 고위험군 - 신속한 대응 필요',
                'urgent': '🆘 긴급 - 즉시 의료진 상담 필요'
            };
            
            priorityBanner.textContent = priorityTexts[recommendations.priority];
            priorityBanner.className = `priority-banner priority-${recommendations.priority}`;
            
            // AI vs 의사 진단 비교 표시
            const aiVsDoctor = recommendations.ai_vs_doctor;
            document.getElementById('aiDiagnosisResult').textContent = aiVsDoctor.ai_diagnosis;
            document.getElementById('aiConfidence').textContent = `(신뢰도: ${aiVsDoctor.ai_confidence}%)`;
            document.getElementById('doctorDiagnosisResult').textContent = aiVsDoctor.doctor_diagnosis;
            
            const agreementStatus = document.getElementById('agreementStatus');
            if (aiVsDoctor.agreement) {
                agreementStatus.textContent = '✅ AI와 의사 진단이 일치합니다';
                agreementStatus.className = 'agreement-status agreement-true';
            } else {
                agreementStatus.textContent = '⚠️ AI와 의사 진단이 다릅니다 (의사 진단을 우선적용)';
                agreementStatus.className = 'agreement-status agreement-false';
            }
            
            // 의사 소견 표시
            const doctorNotesDisplay = document.getElementById('doctorNotesDisplay');
            if (recommendations.doctor_notes) {
                doctorNotesDisplay.innerHTML = `<strong>의사 소견:</strong> ${recommendations.doctor_notes}`;
                doctorNotesDisplay.style.display = 'block';
            } else {
                doctorNotesDisplay.style.display = 'none';
            }
            
            // 예약 정보 표시
            if (appointment && appointment.date) {
                const appointmentCard = document.getElementById('appointmentCard');
                const appointmentDetails = document.getElementById('appointmentDetails');
                
                appointmentDetails.textContent = appointment.message;
                appointmentCard.style.display = 'block';
            }
            
            // 각 카테고리별 권고사항 표시
            const categories = [
                { id: 'medicalAdvice', key: 'medical_advice' },
                { id: 'smokingAdvice', key: 'smoking_advice' },
                { id: 'exerciseAdvice', key: 'exercise_advice' },
                { id: 'lifestyleAdvice', key: 'lifestyle_advice' },
                { id: 'followupAdvice', key: 'followup_advice' }
            ];
            
            categories.forEach(category => {
                const adviceList = recommendations[category.key] || [];
                const listElement = document.querySelector(`#${category.id} .recommendation-list`);
                
                if (adviceList.length > 0) {
                    listElement.innerHTML = '';
                    adviceList.forEach(advice => {
                        const li = document.createElement('li');
                        li.textContent = advice;
                        listElement.appendChild(li);
                    });
                    document.getElementById(category.id).style.display = 'block';
                } else {
                    document.getElementById(category.id).style.display = 'none';
                }
            });
            
            // 권고사항 섹션 표시 및 스크롤
            document.getElementById('recommendationsSection').style.display = 'block';
            setTimeout(() => {
                document.getElementById('recommendationsSection').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 300);
        }
    </script>
</body>
</html>
"""

# 템플릿 폴더 생성 및 HTML 파일 저장
template_dir = 'templates'
os.makedirs(template_dir, exist_ok=True)
with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    if model is None:
        print("❌ 모델을 로드할 수 없어 서버를 시작할 수 없습니다.")
    else:
        print("🚀 서버 시작 중...")
        print("📝 개선된 전처리 알고리즘이 적용되었습니다.")
        print("🎯 Grad-CAM이 원본 이미지에 오버레이됩니다.")
        print("⚠️  전처리 실패 시 관리자 알림 기능이 활성화되었습니다.")
        print("🩺 의사 진단 입력 기능이 추가되었습니다.")
        print("📋 설문조사 및 예약 기능이 추가되었습니다.")
        print("💡 주요 개선사항:")
        print("   - AI 예측 → 의사 진단 → 설문조사 → 권고사항 + 예약완료 워크플로우")
        print("   - AI vs 의사 진단 비교 및 일치/불일치 표시")
        print("   - 의사 소견 및 다음 진료 필요성 판단")
        print("   - 진료 예약 일자 선택 및 예약 완료 확인")
        print("   - 개인화된 건강 권고사항 (의사 진단 반영)")
        print("   - 우선순위별 시각적 구분 (저/중/고/긴급)")
        # 실제 배포 시에는 host를 '0.0.0.0'으로 설정
        app.run(host='localhost', port=8001, debug=False)