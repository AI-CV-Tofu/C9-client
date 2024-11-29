import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import time

# 페이지별 함수 정의
def main_screen():
    st.title("메인 화면")
    st.write("두부 결함 검출 페이지입니다.")
    
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4", "avi", "mov"])
    if uploaded_file:
        st.write("업로드된 영상을 처리합니다...")
        # 비디오 처리 로직 추가
        time.sleep(1)
        st.success("영상 처리 완료!")

def dashboard():
    st.title("대시보드")
    st.write("실시간 데이터 및 통계를 확인하세요.")
    # 대시보드 데이터 표시 (예제 데이터 사용)
    time.sleep(1)
    st.success("대시보드 로딩 완료!")

# 멀티스레딩으로 두 페이지 동작
def run_pages():
    with ThreadPoolExecutor() as executor:
        future_main = executor.submit(main_screen)
        future_dashboard = executor.submit(dashboard)
        # 결과 대기 및 출력
        future_main.result()
        future_dashboard.result()

# Streamlit 사이드바에서 페이지 선택
choice = st.sidebar.radio("메뉴", ["전체 보기", "메인 화면", "대시보드"])

# 선택에 따른 페이지 실행
if choice == "메인 화면":
    main_screen()
elif choice == "대시보드":
    dashboard()
elif choice == "전체 보기":
    run_pages()
