import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
import joblib
import requests
import os
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode, MediaStreamConstraints, AudioProcessorBase
from streamlit_drawable_canvas import st_canvas
import base64
import random # 더미 모델용
from PIL import Image
import io
from typing import List, Union
import av # media 파일 처리 라이브러리 (streamlit-webrtc와 함께 사용됨)
from pydub import AudioSegment # pydub을 사용하여 audio segment 처리 및 WAV 저장
import threading # 오디오 데이터 처리를 위한 스레딩

import warnings
warnings.filterwarnings('ignore')

# --- 0. 전역 설정 및 API 클라이언트 초기화 ---
try:
    client = OpenAI(api_key=st.secrets["openai_api_key"])
except Exception:
    client = None
    st.error("⚠️ OpenAI API 키 설정이 필요합니다. '.streamlit/secrets.toml' 파일을 확인하거나 배포 환경에서 설정해주세요.")

# 모델/스케일러 파일 경로
MODEL_FILE = 'final_model.joblib'
SCALER_FILE = 'final_scaler.joblib'

# --- 1. 모델 및 스케일러 로딩 ---
@st.cache_resource
def load_model_and_scaler():
    """실제 학습된 모델과 스케일러를 로드합니다."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        st.success("✅ 모델과 스케일러 로드 완료.")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ 'final_model.joblib' 또는 'final_scaler.joblib' 파일이 없습니다. 더미 모델을 사용합니다.")
        # 더미 모델 (실제 배포 시 이 부분 삭제)
        class DummyModel:
            def predict(self, X): 
                # 데이터셋의 타겟이 Dem/CN 이었다고 가정
                return np.array([random.choice(['Dem', 'CN'])]) 
            def predict_proba(self, X): return np.array([[0.5, 0.5]])
        class DummyScaler:
            def transform(self, X): return X
        return DummyModel(), DummyScaler()

model, scaler = load_model_and_scaler()

# --- 2. 채점 로직 함수 ---

def get_current_korean_season(month):
    """현재 월에 따른 한국 계절 반환 (간소화)"""
    if month in [3, 4, 5]: return "봄"
    elif month in [6, 7, 8]: return "여름"
    elif month in [9, 10, 11]: return "가을"
    else: return "겨울"

def score_time_date(q_num, user_input, current_dt):
    """Q01 ~ Q05 시간/날짜 지남력 채점 로직"""
    score = 0
    if q_num == 1: # 연도
        if user_input == str(current_dt.year): score = 1
    elif q_num == 2: # 계절
        if user_input == get_current_korean_season(current_dt.month): score = 1
    elif q_num == 3: # 일 (날짜)
        if user_input == str(current_dt.day): score = 1
    elif q_num == 4: # 요일
        korean_day = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][current_dt.weekday()]
        if user_input == korean_day: score = 1
    elif q_num == 5: # 월
        if user_input == str(current_dt.month): score = 1
    return score

@st.cache_data
def get_user_location():
    """공개 IP 기반으로 사용자 위치 (국가, 도시)를 가져옵니다."""
    try:
        ip_response = requests.get('https://api.ipify.org?format=json')
        public_ip = ip_response.json()['ip']
        location_response = requests.get(f'https://ipapi.co/{public_ip}/json/')
        data = location_response.json()
        country = "대한민국" if data.get('country_code') == 'KR' else data.get('country_name', '알 수 없음')
        city = data.get('city', '알 수 없음')
        return country, city
    except Exception:
        return "알 수 없음", "알 수 없음"

def score_stt_response(audio_file_path, target_keywords=None, model_to_use="whisper-1"):
    """
    OpenAI Whisper API를 사용하여 오디오 파일의 텍스트를 추출하고 키워드 포함 여부를 채점합니다.
    주의: audio_file_path는 실제 오디오 파일 경로여야 합니다. (이 코드에서는 더미 경로 사용)
    """
    if client is None:
        return 0, "STT API 클라이언트 오류"
    if not audio_file_path or not os.path.exists(audio_file_path):
        return 0, f"STT: 오디오 파일 없음 또는 경로 오류: {audio_file_path}"
        
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model_to_use, 
                file=audio_file, 
                language="ko"
            ).text.lower()
            
        if target_keywords: # Q18
            if any(keyword.lower() in transcript for keyword in target_keywords):
                score = 1
            else:
                score = 0
            return score, transcript
        else: # Q15: 텍스트가 추출되면 성공으로 간주 (간소화)
            return 1, transcript
        
    except Exception as e:
        return 0, f"STT 처리 오류: {e}"

def score_llm_writing(writing_text):
    """OpenAI GPT 모델을 사용하여 글쓰기 점수(0 또는 1)를 부여합니다. (Q19)"""
    if client is None or not writing_text:
        return 0
        
    system_prompt = ("당신은 인지 기능 평가 전문가입니다. 사용자 글이 주어진 주제('날씨 또는 기분')에 대해 '하나의 온전한 문장'인지 판단하고, "
                    "온전한 문장이면 '1', 아니면 '0'을 출력하세요. 다른 설명은 일절 포함하지 마세요. (예: '비가 온다'는 1점, '비'는 0점)")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"사용자 글: '{writing_text}'"}
            ],
            temperature=0
        )
        score_match = re.search(r'[01]', response.choices[0].message.content)
        return int(score_match.group(0)) if score_match else 0
        
    except Exception:
        return 0

def score_registration_recall(user_input, target_words):
    """Q11, Q13 등록/기억 회상 채점 로직 (순서 무관, 3개 모두 포함)"""
    user_words = set(re.findall(r'\b\w+\b', user_input.replace(',', ' ').lower()))
    target_set = set(target_words)
    return 1 if target_set.issubset(user_words) else 0

def score_drawing_similarity(original_image_url, user_drawing_data_url):
    """
    OpenAI Vision API를 사용하여 원본 그림과 사용자가 그린 그림의 유사도를 평가합니다. (Q17)
    """
    if client is None:
        return 0, "Vision API 클라이언트 오류"

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Vision 기능이 있는 모델
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "첫 번째 그림(원본)을 두 번째 그림(사용자가 그린 그림)이 얼마나 잘 모사했는지 평가하고, '1' (매우 유사) 또는 '0' (유사하지 않음)으로만 답해주세요. 다른 설명은 하지 마세요."},
                        {"type": "image_url", "image_url": {"url": original_image_url, "detail": "low"}},
                        {"type": "image_url", "image_url": {"url": user_drawing_data_url, "detail": "low"}},
                    ],
                }
            ],
            max_tokens=1, # 0 또는 1만 받도록 제한
            temperature=0,
        )
        score_match = re.search(r'[01]', response.choices[0].message.content)
        return int(score_match.group(0)) if score_match else 0, "Vision API 평가 완료"
    except Exception as e:
        return 0, f"Vision API 처리 오류: {e}"


# Lock for thread safety during audio writing
audio_data_lock = threading.Lock()
# 녹음 데이터를 저장할 버퍼
audio_buffers = {} 

class MyAudioProcessor(AudioProcessorBase):
    """
    WebRTC로부터 오디오 프레임을 받아 버퍼에 저장하는 프로세서.
    """
    def __init__(self) -> None:
        self.samples = []
        self.lock = threading.Lock()

    def recv(self, frame: av.AudioFrame) -> Union[av.AudioFrame, None]:
        # 오디오 프레임을 samples 리스트에 추가
        new_samples = frame.to_ndarray(format="s16", layout="mono")
        with self.lock:
            self.samples.append(new_samples)
        
        # 프레임을 그대로 반환하여 연결 유지
        return frame

def st_webrtc_audio_recorder(key, component_label):
    """
    실제 오디오 데이터 저장을 위한 WebRTC 컴포넌트입니다.
    """
    audio_path_key = f"{key}_audio_path"
    
    # 세션 상태가 없으면 초기화
    if audio_path_key not in st.session_state:
        st.session_state[audio_path_key] = None

    # AudioProcessor를 사용하여 스트림을 받아옵니다.
    webrtc_ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV, # 오디오를 받고 보냄
        media_stream_constraints=MediaStreamConstraints(video=False, audio=True),
        audio_processor_factory=MyAudioProcessor, # 사용자 정의 프로세서 사용
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # 오디오 데이터 처리 및 저장 버튼
    if webrtc_ctx.audio_processor:
        processor = webrtc_ctx.audio_processor
        
        # 녹음 데이터가 있다면 저장 버튼 활성화
        if st.button(f"저장 및 채점 ({component_label})", key=f"{key}_save_btn"):
            st.warning("처리 중입니다. 잠시 기다려주세요...")
            
            # 1. 버퍼에서 데이터 추출
            with processor.lock:
                all_samples = np.concatenate(processor.samples, axis=0) if processor.samples else None
                processor.samples = [] # 데이터 추출 후 버퍼 초기화
            
            if all_samples is None or all_samples.size == 0:
                st.error("녹음된 오디오 데이터가 없습니다.")
                return None

            # 2. NumPy 배열을 pydub AudioSegment로 변환 및 저장
            try:
                # s16 (16비트 정수)로 가정
                audio_segment = AudioSegment(
                    all_samples.tobytes(), 
                    frame_rate=48000, # WebRTC 기본 샘플링 속도 (환경에 따라 다를 수 있음)
                    sample_width=all_samples.dtype.itemsize, 
                    channels=1
                )
                
                temp_audio_file = f"uploaded_{key}_{datetime.datetime.now().strftime('%M%S')}.wav"
                audio_segment.export(temp_audio_file, format="wav")
                
                st.session_state[audio_path_key] = temp_audio_file
                st.success(f"✅ 오디오 파일 저장 완료: {temp_audio_file}")
                return temp_audio_file
                
            except Exception as e:
                st.error(f"오디오 저장 실패: {e}")
                return None
    
    return st.session_state.get(audio_path_key)

# --- 3. Streamlit UI 구성 ---

def app():
    st.set_page_config(page_title="MMSE 간이 자가 진단 웹사이트", layout="wide")
    st.title("🧠 MMSE 기반 간이 자가 진단 (고급 시뮬레이션)")
    st.markdown("---")
    
    current_dt = datetime.datetime.now()
    user_country, user_city = get_user_location()

    # Q11/Q13 단어 리스트
    target_words = {"사과", "세탁기", "책상"}
    
    # 세션 상태 초기화
    if 'features' not in st.session_state:
        st.session_state.features = {k: 0 for k in ['Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08', 'Q09', 'Q10', 'Q11_1', 'Q11_2', 'Q11_3', 'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5', 'Q13_1', 'Q13_2', 'Q13_3', 'Q14_1', 'Q14_2', 'Q15', 'Q16_1', 'Q16_2', 'Q16_3', 'Q17', 'Q18', 'Q19']}
        st.session_state.basic_info = {'SAMPLE_EMAIL': '', 'DIAG_SEQ': 1, 'MMSE_KIND': 2}
        st.session_state.q15_audio_path = None
        st.session_state.q18_audio_path = None
        st.session_state.q17_drawing_data_url = None

    with st.form(key='diagnosis_form'):
        
        # --- 기본 정보 ---
        st.header("👤 기본 정보")
        col1, col2 = st.columns(2)
        with col1:
            email = st.text_input("SAMPLE_EMAIL", key='email_input')
            st.session_state.basic_info['SAMPLE_EMAIL'] = email
        with col2:
            st.text_input("DIAG_SEQ", value="1", disabled=True)
            st.text_input("MMSE_KIND", value="2 (고정)", disabled=True)
        st.markdown("---")
        
        # --- 지남력 (시간 & 장소) ---
        st.header("🕰️ 지남력")
        st.markdown("##### 시간")
        q_cols = st.columns(5)
        q01 = q_cols[0].text_input("Q01: 연도", key='q01')
        st.session_state.features['Q01'] = score_time_date(1, q01, current_dt)
        q02 = q_cols[1].text_input("Q02: 계절", key='q02')
        st.session_state.features['Q02'] = score_time_date(2, q02, current_dt)
        q03 = q_cols[2].text_input("Q03: 일", key='q03')
        st.session_state.features['Q03'] = score_time_date(3, q03, current_dt)
        q04 = q_cols[3].text_input("Q04: 요일", key='q04')
        st.session_state.features['Q04'] = score_time_date(4, q04, current_dt)
        q05 = q_cols[4].text_input("Q05: 월", key='q05')
        st.session_state.features['Q05'] = score_time_date(5, q05, current_dt)
        
        st.markdown("##### 장소")
        q_cols = st.columns(5)
        q06 = q_cols[0].text_input(f"Q06: 국가 ({user_country})", key='q06')
        st.session_state.features['Q06'] = 1 if q06 == user_country else 0
        q07 = q_cols[1].text_input(f"Q07: 도시 ({user_city})", key='q07')
        st.session_state.features['Q07'] = 1 if q07 == user_city else 0
        q08 = q_cols[2].text_input("Q08: 건물 유형", key='q08')
        st.session_state.features['Q08'] = 1 if q08 else 0
        q09 = q_cols[3].text_input("Q09: 건물 이름", key='q09')
        st.session_state.features['Q09'] = 1 if q09 else 0
        q10 = q_cols[4].text_input("Q10: 층수", key='q10')
        st.session_state.features['Q10'] = 1 if q10 else 0
        st.markdown("---")
        
        # --- Q11, Q13 등록/회상 ---
        st.header("🍎 기억 등록 및 회상")
        st.info(f"**목표 단어:** {', '.join(target_words)}")
        q11_input = st.text_input("Q11: 세 가지 물건 이름을 따라 말씀해보세요.", key='q11_input')
        q11_score = score_registration_recall(q11_input, target_words)
        st.session_state.features['Q11_1'] = st.session_state.features['Q11_2'] = st.session_state.features['Q11_3'] = q11_score
        st.caption(f"Q11 채점 결과: {q11_score}점 (총 1점)")
        
        q13_input = st.text_input("Q13: 아까 외운 단어 세 가지를 입력하세요.", key='q13_input')
        q13_score = score_registration_recall(q13_input, target_words)
        st.session_state.features['Q13_1'] = st.session_state.features['Q13_2'] = st.session_state.features['Q13_3'] = q13_score
        st.caption(f"Q13 채점 결과: {q13_score}점 (총 1점)")
        st.markdown("---")

        # --- Q12 주의 집중 및 계산 ---
        st.header("🔢 주의 집중 및 계산 (Q12)")
        st.info("100에서 7을 연속해서 빼보세요. (5단계)")
        q12_answers = [93, 86, 79, 72, 65]
        q12_cols = st.columns(5)
        for i, answer in enumerate(q12_answers):
            with q12_cols[i]:
                q12_input = st.number_input(f"Q12_{i+1}: {100 - (i) * 7} - 7 = ?", key=f'q12_{i+1}', step=1, value=None, format="%d")
                score = 1 if q12_input and int(q12_input) == answer else 0
                st.session_state.features[f'Q12_{i+1}'] = score
                st.caption(f"({score}점)")
        st.markdown("---")
        
        # --- Q14~Q19 언어 능력 및 기타 ---
        st.header("🗣️ 언어 및 실행 능력")
        
        # Q14 (이름 대기)
        st.subheader("Q14: 이름 대기")
        st.image("https://i.imgur.com/vH0k1tM.png", caption="원본 이미지: 시계", width=100) 
        q14_1 = st.text_input("Q14_1: 이것은 무엇입니까?", key='q14_1')
        st.session_state.features['Q14_1'] = 1 if '시계' in q14_1 else 0
        
        st.image("https://i.imgur.com/kS5x0N9.png", caption="원본 이미지: 연필", width=100) 
        q14_2 = st.text_input("Q14_2: 이것은 무엇입니까?", key='q14_2')
        st.session_state.features['Q14_2'] = 1 if '연필' in q14_2 else 0

        # Q15 (따라 말하기 - STT 구현, 폼 내부)
        st.subheader("Q15: 따라 말하기 (마이크)")
        st.markdown("'_간장 공장 공장장_'을 마이크에 대고 말해보세요.")
        q15_audio_file_path = st_webrtc_audio_recorder("Q15", "Q15 따라 말하기")
        if q15_audio_file_path:
            st.session_state.q15_audio_path = q15_audio_file_path
        st.markdown("---")
            
        # Q16 (3단계 명령)
        st.subheader("Q16: 3단계 명령 수행")
        st.text("종이를 드리면, 종이를 뒤집은 다음, 반으로 접어서 저에게 주세요.")
        q16_1 = st.checkbox("Q16_1: 종이를 뒤집었습니다.", key='q16_1')
        q16_2 = st.checkbox("Q16_2: 반으로 접었습니다.", key='q16_2')
        q16_3 = st.checkbox("Q16_3: 저에게 주었습니다.", key='q16_3')
        st.session_state.features['Q16_1'] = 1 if q16_1 else 0
        st.session_state.features['Q16_2'] = 1 if q16_2 else 0
        st.session_state.features['Q16_3'] = 1 if q16_3 else 0

        # Q17 (따라 그리기 - Canvas + OpenAI Vision API)
        st.subheader("Q17: 따라 그리기")
        q17_original_image_url = "https://i.imgur.com/gK9p5Fz.png" # 원본 그림 URL
        st.image(q17_original_image_url, caption="원본 그림: 오각형과 사각형", width=150)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0)", # 채우기 색 없음
            stroke_width=3, # 펜 두께 조절 (기본값 3으로 설정)
            stroke_color="#000000", # 검은색 펜
            background_image=None,
            width=250,
            height=250,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # 그린 그림 데이터를 base64 Data URL로 저장 (제출 시 Vision API 호출)
        if canvas_result.image_data is not None:
            # 🔥🔥🔥 수정된 부분: NumPy 배열을 PIL Image 객체로 변환 🔥🔥🔥
            image_array = canvas_result.image_data # NumPy ndarray
            
            # RGBA 배열을 RGB로 변환 (PIL Image.fromarray는 RGBA도 처리 가능)
            pil_image = Image.fromarray(image_array.astype('uint8'), 'RGBA') 
            
            # 2. PIL Image 객체의 save 메서드 사용
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG") # PIL Image 객체는 save() 메서드를 가짐
            
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.session_state.q17_drawing_data_url = f"data:image/png;base64,{img_str}"
        else:
            st.session_state.q17_drawing_data_url = None
        st.markdown("---")

        # Q18 (읽고 수행 - STT 구현, 폼 내부)
        st.subheader("Q18: 문장 읽고 수행 (마이크)")
        st.markdown("'_눈을 감으세요_' 문장을 마이크에 대고 읽은 후, 눈을 감는 시늉을 해보세요.")
        q18_audio_file_path = st_webrtc_audio_recorder("Q18", "Q18 읽고 수행")
        if q18_audio_file_path:
            st.session_state.q18_audio_path = q18_audio_file_path
        st.markdown("---")

        # Q19 (글쓰기 - LLM 채점)
        st.subheader("Q19: 문장 만들기")
        st.text("여기에 날씨(또는 기분)에 대해서 한 문장으로 써 보세요.")
        q19_text = st.text_area("Q19: 문장을 입력하세요.", key='q19')
        st.markdown("---")
        
        submit_button = st.form_submit_button(label='자가 진단 결과 확인 및 예측')

    # --- 4. 제출 및 예측 ---
    if submit_button:
        # 1. STT 최종 채점 (Q15, Q18)
        q15_score, q15_transcript = score_stt_response(st.session_state.q15_audio_path, target_keywords=None)
        st.session_state.features['Q15'] = q15_score
        
        q18_score, q18_transcript = score_stt_response(st.session_state.q18_audio_path, target_keywords=["눈을 감으세요"]) 
        st.session_state.features['Q18'] = q18_score
        
        # 2. Q17 Vision API 채점
        q17_score, q17_vision_status = 0, "그린 그림 없음"
        if st.session_state.q17_drawing_data_url:
            q17_score, q17_vision_status = score_drawing_similarity(q17_original_image_url, st.session_state.q17_drawing_data_url)
        st.session_state.features['Q17'] = q17_score
        
        # 3. LLM 최종 채점 (Q19)
        q19_score = score_llm_writing(q19_text)
        st.session_state.features['Q19'] = q19_score
        
        # 4. 모델 입력 준비
        features_for_model = {**st.session_state.basic_info, **st.session_state.features}
        features_for_model.pop('SAMPLE_EMAIL')
        
        feature_order = [
            'DIAG_SEQ', 'MMSE_KIND', 
            'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08', 'Q09', 'Q10',
            'Q11_1', 'Q11_2', 'Q11_3', 
            'Q12_1', 'Q12_2', 'Q12_3', 'Q12_4', 'Q12_5', 
            'Q13_1', 'Q13_2', 'Q13_3', 
            'Q14_1', 'Q14_2', 
            'Q15', 'Q16_1', 'Q16_2', 'Q16_3', 'Q17', 'Q18', 'Q19'
        ]
        
        input_data = {k: features_for_model[k] for k in feature_order}
        input_df = pd.DataFrame([input_data])

        # 5. 예측 수행
        if model is not None and scaler is not None:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            result_text = "Dem (치매/위험군)" if prediction[0] == 'Dem' else "CN (정상)" 
        else:
            result_text = "모델 로드 실패"

        st.subheader("✅ 자가 진단 결과")
        if "Dem" in result_text:
            st.error(f"**진단 결과:** {result_text}")
            st.warning("⚠️ 이 결과는 치매 위험군(MCI/Dem)으로 분류된 시뮬레이션입니다. 전문의의 진료를 반드시 받으십시오.")
        elif "CN" in result_text:
            st.success(f"**진단 결과:** {result_text}")
            st.info("이 결과는 간이 진단이며, 정기적인 검진이 필요합니다.")
        else:
            st.error(result_text)
        
        st.subheader("📊 채점 결과 요약")
        # MMSE 총점 (DIAG_SEQ, MMSE_KIND를 제외한 Q 점수 합산)
        total_score = input_df.iloc[0].drop(['DIAG_SEQ', 'MMSE_KIND']).sum()
        st.metric(label="총 점수 (Max 26점)", value=f"{total_score}점")
        
        st.caption(f"Q15 (따라 말하기) 채점: {q15_score}점 (STT 전사: {q15_transcript[:50]}...)")
        st.caption(f"Q18 (읽고 수행) 채점: {q18_score}점 (STT 전사: {q18_transcript[:50]}...)")
        st.caption(f"Q17 (따라 그리기) 채점: {q17_score}점 ({q17_vision_status})")
        st.caption(f"Q19 (글쓰기) 채점: {q19_score}점 (LLM 판정)")
        
        st.dataframe(input_df.T.rename(columns={0: '입력 피처 값'}), use_container_width=True)


if __name__ == "__main__":
    app()