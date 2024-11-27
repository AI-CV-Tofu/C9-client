import requests
import streamlit as st
import json

def fetch_data_from_api():
    url = "http://44.214.252.225:8000/dashboard"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("API 호출 실패: " + str(response.status_code))
            return None
    except Exception as e:
        st.error(f"API 호출 중 오류 발생: {e}")
        return None
        
        
def fetch_data_stream(url):
    try:
        with requests.get(url, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        data = json.loads(decoded_line)
                        print(data)
                        st.session_state['dashboard_data'] = data
                        st.experimental_rerun()
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        st.error(f"Stream error: {e}")