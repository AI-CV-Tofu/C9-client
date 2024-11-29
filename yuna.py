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
import queue
import threading
import requests
import os

# 세션 상태 초기화
if "video_processing_queue" not in st.session_state:
    st.session_state.video_processing_queue = queue.Queue()

if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

if "processing_thread" not in st.session_state:
    st.session_state.processing_thread = None


def send_image_as_numpy_array(image):
    """
    이미지를 numpy 배열로 변환하여 서버로 전송
    """
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
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def apply_tint(image, status):
    """
    이미지에 상태에 따라 색상 적용
    """
    overlay = image.copy()
    output = image.copy()

    tint_color = (0, 255, 0) if status == "OK" else (0, 0, 255)
    alpha = 0.1
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), tint_color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def is_tofu_fully_visible(frame, x, y, w, h):
    """두부 완전 검출 조건 확인"""
    height, width, _ = frame.shape
    aspect_ratio = w / h
    tofu_area = w * h
    frame_area = width * height

    return (
        tofu_area >= frame_area * 0.9 and  
        0.971 <= aspect_ratio <= 1.1 and    
        0 <= x and 0 <= y and x + w <= width and y + h <= height
    )


def detect_best_tofu_frame(frame, frame_center):
    """두부 검출 및 중앙 조건 확인"""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray_frame, 50, 170)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if is_tofu_fully_visible(frame, x, y, w, h):
            tofu_center = (x + w / 2, y + h / 2)
            distance_to_center = sqrt((tofu_center[0] - frame_center[0])**2 + (tofu_center[1] - frame_center[1])**2)
            if distance_to_center < frame.shape[1] * 0.05:  # 중앙 근접 조건
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def video_processing_task(uploaded_file, result_queue):
    """
    동영상 처리 작업 실행 (백그라운드)
    """
    try:
        temp_file_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            result_queue.put({"error": "동영상을 열 수 없습니다."})
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

            result_queue.put({
                "frame_idx": frame_idx + 1,
                "total_frames": total_frames,
                "frame": detected_frame if detected_frame is not None else frame_resized,
            })

        cap.release()

    except Exception as e:
        result_queue.put({"error": str(e)})
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def start_video_processing(uploaded_file):
    """
    동영상 처리 시작
    """
    if st.session_state.processing_thread is not None and st.session_state.processing_thread.is_alive():
        st.info("이미 동영상 처리가 진행 중입니다.")
        return

    if uploaded_file:
        st.session_state.processing_thread = threading.Thread(
            target=video_processing_task,
            args=(uploaded_file, st.session_state.result_queue),
            daemon=True
        )
        st.session_state.processing_thread.start()
        st.success("동영상 처리가 시작되었습니다.")


def display_video_processing_ui():
    """
    동영상 처리 결과 표시
    """
    while not st.session_state.result_queue.empty():
        result = st.session_state.result_queue.get()
        if "error" in result:
            st.error(result["error"])
        else:
            video_placeholder.image(
                cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB),
                caption=f"프레임 {result['frame_idx']}/{result['total_frames']}",
                width=400,
            )


# 페이지 선택 사이드바 생성
with st.sidebar:
    choice = option_menu("Menu", ["메인 화면", "대시보드"],
                         icons=['house', 'bi bi-clipboard2-data'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             "container": {"padding": "4!important"},
                             "icon": {"font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                             "nav-link-selected": {"background-color": "#90c43c"},
                         })

# 메인 화면
if choice == "메인 화면":  # 메뉴 선택 조건
    with st.sidebar:
        uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4", "avi", "mov"])
    st.markdown(
        """
        <style>
        div.block-container {
            max-width: 90%;  /* 대시보드 너비 설정 */
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
        margin-left: -270px; /* col2만 왼쪽으로 이동 */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("두부 결함 검출")

    # 넓은 화면 레이아웃 설정
    col1, col2 = st.columns([2, 1])
    # 기본 UI 초기화
    col1.write("영상")
    video_placeholder = col1.empty()  # 비디오 자리
    col2.write("현재 두부 사진")
    tofu_image_placeholder = col2.empty()  # 두부 이미지 자리
    
    status_placeholder = col2.empty()
    defect_info_placeholder = col2.empty()
    

    # 영상 미 업로드 시 기본 UI 표시
    light_green_color = (169, 209, 150)  # RGB 값
    default_image = np.full((400, 400, 3), light_green_color, dtype=np.uint8)

    video_placeholder.image(default_image, caption="영상 없음", width=400)
    tofu_image_placeholder.image(default_image, caption="두부 이미지 없음", width=400)

    if uploaded_file:
        start_video_processing(uploaded_file)
    display_video_processing_ui()



# 대시보드
elif choice == "대시보드":
    api_data = fetch_data_from_api()
    # print(api_data)
    st.markdown(
        """
        <style>
        div.block-container {
            max-width: 90%;  /* 대시보드 너비 설정 */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("대시보드")

    # 상단 3개의 원형 차트 영역
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("공장 1")
        fig1 = px.pie(
            values=api_data['data']['pie_chart'], 
            names=["OK", "NG"], 
            hole=0.4,
            color=["OK", "NG"],
            color_discrete_map={"OK": "lightblue", "NG": "pink"}
            )
        fig1.update_traces(hovertemplate="")
        st.plotly_chart(fig1, use_container_width=True)

        # Expander - 공장 1 NG 목록 (비율)
        with st.expander("공장 1 NG 목록 (비율)"):
            ng_data1 = pd.DataFrame({
                "클래스": ["선자국", "이물질", "잔재", "패임", "절단면", "모서리 깨짐", "기포"],
                "비율 (%)": [30, 20, 15, 10, 10, 10, 5]
            })
            fig_ng1 = px.bar(ng_data1, x="비율 (%)", y="클래스", orientation="h", title="공장 1 NG 검출 비율")
            fig_ng1.update_layout(
                yaxis_title="",  # y축 레이블 제거
                yaxis=dict(
                    tickfont=dict(size=12),  # y축 폰트 크기 유지
                    automargin=True  # y축 자동 마진 활성화
                ),
                margin=dict(l=70, r=20, t=30, b=20)  # 마진 설정
            )
            st.plotly_chart(fig_ng1, use_container_width=True)

    with col2:
        st.subheader("공장 2")
        fig2 = px.pie(
            values=[70, 30], 
            names=["OK", "NG"], 
            hole=0.4,
            color=["OK", "NG"],
            color_discrete_map={"OK": "lightblue", "NG": "pink"}
            )
        fig2.update_traces(hovertemplate="")
        st.plotly_chart(fig2, use_container_width=True)

        # Expander - 공장 2 NG 목록 (비율)
        with st.expander("공장 2 NG 목록 (비율)"):
            ng_data2 = pd.DataFrame({
                "클래스": ["선자국", "이물질", "잔재", "패임", "절단면", "모서리 깨짐", "기포"],
                "비율 (%)": [25, 25, 20, 15, 5, 5, 5]
            })
            fig_ng2 = px.bar(ng_data2, x="비율 (%)", y="클래스", orientation="h", title="공장 2 NG 검출 비율")
            fig_ng2.update_layout(
                yaxis_title="",
                yaxis=dict(
                    tickfont=dict(size=12),
                    automargin=True
                ),
                margin=dict(l=70, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_ng2, use_container_width=True)

    with col3:
        st.subheader("공장 3")
        fig3 = px.pie(
            values=[60, 40], 
            names=["OK", "NG"], 
            hole=0.4,
            color=["OK", "NG"],
            color_discrete_map={"OK": "lightblue", "NG": "pink"}
            )
        fig3.update_traces(hovertemplate="")
        st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

        # Expander - 공장 3 NG 목록 (비율)
        with st.expander("공장 3 NG 목록 (비율)"):
            ng_data3 = pd.DataFrame({
                "클래스": ["선자국", "이물질", "잔재", "패임", "절단면", "모서리 깨짐", "기포"],
                "비율 (%)": [40, 20, 10, 10, 10, 5, 5]
            })
            fig_ng3 = px.bar(ng_data3, x="비율 (%)", y="클래스", orientation="h", title="공장 3 NG 검출 비율")
            fig_ng3.update_layout(
                yaxis_title="",
                yaxis=dict(
                    tickfont=dict(size=12),
                    automargin=True
                ),
                margin=dict(l=70, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_ng3, use_container_width=True)

    # 실시간 그래프 영역
    st.subheader("실시간 그래프")
    st.write("→ 시간대별 검출 현황 (선 그래프 / 막대 그래프)")

    # 예시 데이터 생성
    data = pd.DataFrame({
        "시간": api_data['data']['line_chart']['timestamp'],
        "OK": api_data['data']['line_chart']['OK'],
        "NG": api_data['data']['line_chart']['NG']
    })

    # 선 그래프 예제
    fig_line = px.line(data, x="시간", y=["OK", "NG"], markers=True, title="시간대별 검출 현황")
    st.plotly_chart(fig_line, use_container_width=True)

