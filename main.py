import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="기온 예측기", layout="centered")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
CUTOFF_YEAR = 2025
MIN_OBS_DAYS = 300


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df


def build_yearly(df):
    yearly = df.groupby("연도").agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count"),
    ).reset_index()

    yearly = yearly[yearly["연도"] <= CUTOFF_YEAR]
    yearly = yearly[yearly["관측일수"] >= MIN_OBS_DAYS]
    yearly = yearly.sort_values("연도").reset_index(drop=True)
    return yearly


st.title("🌡️ 서울 기온 예측기")
st.caption("서울 연평균 기온 데이터를 기반으로 회귀 직선을 만들고, 원하는 연도의 예상 기온을 확인해 보세요.")

with st.spinner("데이터를 불러오는 중입니다..."):
    raw_df = load_data()
    yearly_df = build_yearly(raw_df)

if yearly_df.empty:
    st.error("조건을 만족하는 연도별 데이터가 없습니다.")
    st.stop()

# 회귀 계수 계산
x = yearly_df["연도"].values
y = yearly_df["평균기온"].values
slope, intercept = np.polyfit(x, y, 1)
corr = np.corrcoef(x, y)[0, 1]

n_years = len(yearly_df)
start_year = int(yearly_df["연도"].min())
end_year = int(yearly_df["연도"].max())

# 산점도 + 회귀직선
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers", name="연평균기온",
    marker=dict(color="royalblue", size=8)
))

line_x = np.array([x.min(), x.max()])
line_y = slope * line_x + intercept
fig.add_trace(go.Scatter(
    x=line_x, y=line_y, mode="lines", name="회귀 직선",
    line=dict(color="firebrick", width=2)
))

fig.update_layout(
    title="서울 연평균기온 산점도 및 회귀 직선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(f"**상관계수 (r)**: `{corr:.4f}`")
st.markdown(
    f"**회귀 직선**: 사용한 연도 수 `{n_years}`개 "
    f"(시작 연도: `{start_year}` ~ 끝 연도: `{end_year}`)"
)
st.caption(f"기준: {CUTOFF_YEAR}년까지의 자료 중, 연간 관측일수가 {MIN_OBS_DAYS}일 이상인 연도만 사용했습니다.")

st.divider()

# 연도 슬라이더 및 예측
st.subheader("연도별 예상 기온")
selected_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)
predicted_temp = slope * selected_year + intercept

st.metric(label=f"{selected_year}년 예상 평균기온", value=f"{predicted_temp:.2f} °C")

if selected_year < start_year or selected_year > end_year:
    st.warning("선택한 연도는 실제 데이터 범위를 벗어난 외삽(extrapolation) 값입니다. 참고용으로만 활용하세요.")
