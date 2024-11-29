import cv2
import numpy as np
import streamlit as st
from math import sqrt
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd
import tempfile
import time
from utils import fetch_data_from_api
import requests
import base64
import tempfile
import os
import asyncio
from streamlit_autorefresh import st_autorefresh

# 세션 상태 초기화
if "defect_data" not in st.session_state:
    st.session_state["defect_data"] = {"OK": 0, "NG": 0}

if "last_update" not in st.session_state:
    st.session_state["last_update"] = None


def send_image_as_numpy_array(image):
    """서버로 이미지 데이터 전송"""
    url = "http://44.214.252.225:8000/process-image/"
    try:
        image_array = image.astype(np.uint8).tolist()
        response = requests.post(
            url,
            json={"image": image_array},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: HTTP {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error while sending image: {e}")
        return None


def detect_best_tofu_frame(frame, frame_center):
    """두부 검출 로직"""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray_frame, 50, 170)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 5000:  # 두부 크기 조건
            tofu_center = (x + w // 2, y + h // 2)
            distance_to_center = ((tofu_center[0] - frame_center[0]) ** 2 + (tofu_center[1] - frame_center[1]) ** 2) ** 0.5
            if distance_to_center < 50:  # 중앙 가까움 조건
                return frame[y:y+h, x:x+w]
    return None


async def process_video(uploaded_file):
    """동영상 처리 및 두부 검출"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)
    if not cap.isOpened():
        st.error("동영상을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 640))
        frame_center = (frame_resized.shape[1] // 2, frame_resized.shape[0] // 2)
        detected_frame = detect_best_tofu_frame(frame_resized, frame_center)

        # 두부 검출 및 서버 전송
        if detected_frame is not None:
            response = send_image_as_numpy_array(detected_frame)
            if response and response.get("status") == "success":
                defect_status = response["results"]["defect_status"]

                # 세션 상태 업데이트
                if defect_status == "OK":
                    st.session_state["defect_data"]["OK"] += 1
                elif defect_status == "NG":
                    st.session_state["defect_data"]["NG"] += 1

                # 마지막 업데이트 시간 저장
                st.session_state["last_update"] = time.time()

        await asyncio.sleep(1 / fps)  # 프레임 속도 맞추기


# 탭 생성
tabs = st.tabs(["메인 화면", "대시보드"])

# 메인 화면
with tabs[0]:
    st.title("두부 결함 검출")
    uploaded_file = st.file_uploader("동영상을 업로드하세요", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        asyncio.run(process_video(uploaded_file))

# 대시보드
with tabs[1]:
    st_autorefresh(interval=2000)  # 2초마다 새로고침
    st.title("대시보드")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("검출 상태")
        defect_data = st.session_state["defect_data"]
        st.write(f"OK: {defect_data['OK']}")
        st.write(f"NG: {defect_data['NG']}")

    with col2:
        st.subheader("비율 차트")
        fig = px.pie(
            names=["OK", "NG"],
            values=[defect_data["OK"], defect_data["NG"]],
            color=["OK", "NG"],
            title="검출 비율"
        )
        st.plotly_chart(fig)
