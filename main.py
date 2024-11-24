import cv2
import numpy as np
import streamlit as st
from math import sqrt
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd
import tempfile
import time

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

def process_video(uploaded_file):
    """업로드된 동영상 처리 및 두부 검출"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    cap = cv2.VideoCapture(temp_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, total_frames


# 페이지 선택 사이드바 생성
with st.sidebar:
    choice = option_menu("Menu", ["메인 화면", "대시보드", "과거 기록"],
                         icons=['house', 'bi bi-clipboard2-data', 'bi bi-clock-history'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             "container": {"padding": "4!important"},
                             "icon": {"font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                             "nav-link-selected": {"background-color": "#90c43c"},
                         })

# 메인 화면
if choice == "메인 화면":
    with st.sidebar:  # '메인 화면'에서만 영상 업로드 기능을 추가
        uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4", "avi", "mov"])
    
    st.title("메인 화면")

    # 넓은 화면을 활용한 레이아웃 설정
    col1, col2 = st.columns([2, 1])
    col3, col4, col5 = st.columns([1, 1, 1])

    if uploaded_file is not None:
        # 동영상 처리
        cap, fps, total_frames = process_video(uploaded_file)
        # UI 구성 요소
        col1.write("영상")
        video_placeholder = col1.empty()
        col2.write("현재 두부 사진")
        tofu_image_placeholder = col2.empty()

        col3.write("통계")
          # col3의 바로 아래에 사진 배치
        stat_chart = col3.empty()
        col4.write("결함 위치 크롭")
        col5.write("결함 안내")

        # 초기값 설정
        last_detection_time = -1
        central_frame_image = None
    
        # 프레임 처리
        frame_skip = 5  # 5 프레임마다 처리
        for frame_idx in range(0, total_frames, frame_skip):
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            
            resized_frame = cv2.resize(frame, (300, 300))
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, caption="영상 프레임", width=300)

            frame_center = (resized_frame.shape[1] / 2, resized_frame.shape[0] / 2)
            # 두부 검출
            detected_frame = detect_best_tofu_frame(resized_frame, frame_center)
            # 두부 검출 결과 표시
            if detected_frame is not None and (current_time - last_detection_time > 1):
                central_frame_image = detected_frame
                tofu_image_placeholder.image(
                    detected_frame, caption=f"새로운 두부 검출됨 (Time: {current_time:.2f}s)", width=300
                )
                last_detection_time = current_time

            time.sleep(1/fps)
            
             # 결함 통계 표시 (예시 데이터)
            stat_chart.bar_chart({"불량": [5], "OK": [20]})


        cap.release()

# 대시보드
elif choice == "대시보드":
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
            values=[43, 10], 
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
        "시간": ["10:00", "10:10", "10:20", "10:30", "10:40"],
        "OK": [10, 15, 20, 25, 30],
        "NG": [5, 7, 3, 8, 6]
    })

    # 선 그래프 예제
    fig_line = px.line(data, x="시간", y=["OK", "NG"], markers=True, title="시간대별 검출 현황")
    st.plotly_chart(fig_line, use_container_width=True)

# 과거 기록
elif choice == "과거 기록":
    st.title("과거 기록")