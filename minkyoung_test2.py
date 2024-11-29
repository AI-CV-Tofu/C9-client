import streamlit as st
import asyncio
import aiohttp  # 비동기 HTTP 요청을 처리할 라이브러리
import json

async def fetch_sse_events(url):
    """SSE 이벤트를 수신하는 비동기 함수"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    async for event in response.content:
                        try:
                            # SSE 이벤트 처리
                            event_data = event.decode("utf-8")
                            if event_data.startswith("data:"):
                                event_data = event_data[5:].strip()
                                data = json.loads(event_data)  # JSON 데이터 파싱
                                yield data
                        except json.JSONDecodeError as e:
                            st.error(f"JSON 파싱 오류: {e}")
    except Exception as e:
        st.error(f"SSE 수신 중 오류 발생: {e}")

def update_dashboard_with_sse_data(event_data):
    """대시보드 업데이트 로직"""
    st.write(f"수신된 데이터: {event_data}")
    st.metric(label="수신된 메시지", value=event_data.get("message"))
    st.metric(label="타임스탬프", value=event_data.get("time"))

async def run_sse_client():
    """SSE 클라이언트 실행"""
    url = "http://44.214.252.225:8000/dashboard-stream/"  # SSE 서버 URL
    async for event in fetch_sse_events(url):
        update_dashboard_with_sse_data(event)

# Streamlit UI
st.title("SSE 대시보드")
start_button = st.button("SSE 수신 시작")

if start_button:
    asyncio.run(run_sse_client())
