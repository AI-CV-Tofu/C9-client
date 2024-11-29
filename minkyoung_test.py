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
import os

url = "http://44.214.252.225:8000/process-image/"

def send_image_as_numpy_array(image):
    """
    이미지를 numpy 배열로 변환하여 서버로 전송
    """
    try:
        # 이미지를 numpy 배열로 변환
        image_array = image.astype(np.uint8).tolist()

        # 서버로 전송
        response = requests.post(
            url,
            json={"image": image_array},  # numpy 배열을 리스트로 변환하여 전송
            headers={"Content-Type": "application/json"}
        )

        # 응답 처리
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: HTTP {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error while sending image as numpy array to server: {e}")
        return None

def apply_tint(image, status):
    overlay = image.copy()
    output = image.copy()

    if status == "NG":
        # 노란색 (BGR: Blue, Green, Red)
        tint_color = (0, 0, 255)
    elif status == "OK":
        # 초록색 (BGR: Blue, Green, Red)
        tint_color = (0, 255, 0)  # Green
    else:
        return image  # 상태가 없으면 원본 반환

    # 투명도 설정 (값이 높을수록 원본 이미지에 가까움)
    alpha = 0.1 # 투명도 (0.0 ~ 1.0)

    # 색상 오버레이 추가
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
if choice == "메인 화면":
    with st.sidebar:
        uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4", "avi", "mov"])

    st.title("메인 화면")

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

    if uploaded_file is not None:
        temp_file_path = None  # temp_file_path 초기화
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
    
            # 동영상 읽기
            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                st.error("동영상을 열 수 없습니다. 파일 형식을 확인해주세요.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
                ng_defects = []  # NG 두부 데이터를 저장할 리스트
    
                for frame_idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
    
                    # 프레임 크기 조정 및 중앙 좌표 계산
                    frame_resized = cv2.resize(frame, (640, 640))
                    frame_center = (frame_resized.shape[1] // 2, frame_resized.shape[0] // 2)
                    detected_frame = detect_best_tofu_frame(frame_resized, frame_center)
    
                    # 비디오 프레임 업데이트
                    video_placeholder.image(
                        cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB),
                        caption=f"프레임 {frame_idx + 1}/{total_frames}",
                        width=400
                    )
    
                    # 두부 검출 및 서버로 데이터 전송
                    if detected_frame is not None:
                        response = send_image_as_numpy_array(detected_frame)
                        if response:
                            if response.get("status") == "success":
                                defect_status = response["results"]["defect_status"]
                                defects = response["results"].get("defects", [])
                    
                                # 결함 영역 박스 그리기
                                annotated_frame = detected_frame.copy()
                                image_height, image_width, _ = annotated_frame.shape
                                model_height, model_width = 640, 640
                                x_ratio = image_width / model_width
                                y_ratio = image_height / model_height
                                
                                for defect in defects:
                                    if defect.get("box"):
                                        x1, y1, x2, y2 = map(int, defect["box"])
                                        
                                        x1, x2 = int(x_ratio * x1), int(x_ratio * x2)
                                        y1, y2 = int(y_ratio * y1), int(y_ratio * y2)
                                        
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(annotated_frame, defect["type"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                # 상태 메시지 업데이트
                                if defect_status == "NG":
                                    status_placeholder.warning(f"현재 두부 상태: NG (결함 발견)")
                                else:
                                    status_placeholder.success(f"현재 두부 상태: OK (결함 없음)")
                    
                                # 현재 두부 이미지 업데이트 (결함 영역 포함)
                                tinted_image = apply_tint(annotated_frame, defect_status)
                                tofu_image_placeholder.image(
                                    cv2.cvtColor(tinted_image, cv2.COLOR_BGR2RGB),
                                    caption="현재 두부 이미지 (결함 표시)",
                                    width=400
                                )

                                # 결함 정보 업데이트
                                with defect_info_placeholder:
                                    if defects:
                                        defect_classes = [defect["type"] for defect in defects]
                                        defect_counts = {cls: defect_classes.count(cls) for cls in set(defect_classes)}
                                        summary = "\n>".join([f"{cls}: {count}개" for cls, count in defect_counts.items()])
                                        st.info(summary)
                                    else:
                                        st.info("결함 없음")

                            else:
                                status_placeholder.error(f"서버 에러: {response}")
                        else:
                            status_placeholder.error("서버로부터 응답이 없습니다.")

                cap.release()
    
        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            # 임시 파일 삭제
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# 대시보드
elif choice == "대시보드":
    st.markdown(
        """
        <style>
        div.block-container {
            max-width: 90%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("대시보드")
    
    # 플레이스홀더 생성
    dashboard_placeholder = st.empty()
    
    while True:
        # 현재 타임스탬프를 key에 포함하여 고유성 보장
        current_time = int(time.time())
        
        with dashboard_placeholder.container():
            # 대시보드 데이터 가져오기
            api_data = fetch_data_from_api()

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
                st.plotly_chart(fig1, use_container_width=True, key=f"pie1_{current_time}")

                with st.expander("공장 1 NG 목록 (비율)"):
                    ng_data1 = pd.DataFrame({
                        "클래스": ["기포", "모서리 깨짐", "절단면", "잔재", "패임", "선자국", "이물질"],
                        "비율 (%)": api_data['data']['bar_chart']['counts']
                    })
                    fig_ng1 = px.bar(ng_data1, x="비율 (%)", y="클래스", orientation="h", title="공장 1 NG 검출 비율")
                    fig_ng1.update_layout(
                        yaxis_title="",
                        yaxis=dict(
                            tickfont=dict(size=12),
                            automargin=True
                        ),
                        margin=dict(l=70, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig_ng1, use_container_width=True, key=f"bar1_{current_time}")

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
                st.plotly_chart(fig2, use_container_width=True, key=f"pie2_{current_time}")

                with st.expander("공장 2 NG 목록 (비율)"):
                    ng_data2 = pd.DataFrame({
                        "클래스": ["기포", "모서리 깨짐", "절단면", "잔재", "패임", "선자국", "이물질"],
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
                    st.plotly_chart(fig_ng2, use_container_width=True, key=f"bar2_{current_time}")

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
                st.plotly_chart(fig3, theme="streamlit", use_container_width=True, key=f"pie3_{current_time}")

                with st.expander("공장 3 NG 목록 (비율)"):
                    ng_data3 = pd.DataFrame({
                        "클래스": ["기포", "모서리 깨짐", "절단면", "잔재", "패임", "선자국", "이물질"],
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
                    st.plotly_chart(fig_ng3, use_container_width=True, key=f"bar3_{current_time}")

            # 실시간 그래프 영역
            st.subheader("실시간 그래프")
            st.write("→ 시간대별 검출 현황 (선 그래프 / 막대 그래프)")

            # 데이터 생성
            data = pd.DataFrame({
                "시간": api_data['data']['line_chart']['timestamp'],
                "OK": api_data['data']['line_chart']['OK'],
                "NG": api_data['data']['line_chart']['NG']
            })

            # 선 그래프
            fig_line = px.line(data, x="시간", y=["OK", "NG"], markers=True, title="시간대별 검출 현황")
            st.plotly_chart(fig_line, use_container_width=True, key=f"line_{current_time}")

        # 1초 대기
        time.sleep(1)