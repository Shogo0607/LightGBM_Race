import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import optuna.integration.lightgbm as lgb
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Line,Bar
from pyecharts import options as opts
import pickle
import io
import re

st.set_page_config(page_title="LightGBM")
   
@st.cache(allow_output_mutation=True)
def pickle_model(model):
    f = io.BytesIO()
    pickle.dump(model, f)
    return f

@st.cache()
def read_data(predict_file,model_file):
    predict_df = pd.read_csv(predict_file, encoding="shift-jis")
    loaded_model = pickle.load(model_file)
    return predict_df, loaded_model

def line_chart(x,y,y_name):
    b = (
            Line()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15,)),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(title="",),
            )
        )
    return b

def bar_chart(x,y,y_name):
    b = (
            Bar()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15,)),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(title="",),
            )
        )
    return b
# タイトル
st.title("LightGBM")
st.sidebar.title("LightGBM")

st.header("手順")
# ファイル入力
st.subheader("①予測用CSVファイル")
predict_file = st.sidebar.file_uploader("予測用のCSVファイルを入力してください",type=["csv"])

if not predict_file:
    st.warning("予測用のCSVファイルを入力してください")
    st.stop()
st.success("予測用CSVファイル入力完了")

st.subheader("②学習済みモデル")
model_file = st.sidebar.file_uploader("学習済みモデル(.pkl)を入力してください",type=["pkl"])

if not model_file:
    st.warning("学習済みモデル(.pkl)を入力してください")
    st.stop()
st.success("学習済みモデル(.pkl)入力完了")

# 投票番号入力
st.subheader("③投票番号")
# vote = st.sidebar.selectbox("投票番号を選択してください",("","1,2","3","4","5","6","7","8","9","10","11"))

m = re.findall(r'\d+',model_file.name)
if m[0] == "1":
    vote = "1,2"
else:
    vote = str(m[0])

st.success("投票番号"+vote+"を選択")

st.subheader("④データ読み込み")
with st.spinner("データを読み込んでいます"):
    predict_df, loaded_model = read_data(predict_file,model_file)
st.success("データ読み込み完了")

st.subheader("⑤データ前処理")
with st.spinner("前処理を実施中"):
    predict_df = predict_df[predict_df["馬除外フラグ"] != 1]
st.success("データ前処理完了")

if vote == "1,2":
    predict = predict_df[(predict_df["投票"]==1)|(predict_df["投票"]==2)]
    X_predict = predict[["頭数","生後日数","騎手指数","見習い騎手","種牡馬","父系統","母父系統","父年齢","母年齢","血統距離適性","厩舎ランク","調教師","生産者","馬主","蹄","斤量","斤量構成比","馬体重","予想IDM","コース枠番","奇数偶数","調教素点","厩舎素点","CID素点","CID","調教評価","厩舎評価","追切指数","終いF指数","仕上指数","調教パターン","輸送","外厩","放牧先ランク","騎手期待単勝率","騎手期待連対率","騎手期待複勝率","激走指数","LS指数","万券指数","基準支持率"]]
    y_predict = predict[["単勝配当"]]

elif vote == "3" or vote == "4":
    predict = predict_df[(predict_df["投票"]==int(vote))]
    X_predict = predict[["頭数","生後日数","騎手指数","見習い騎手","種牡馬","父系統","母父系統","父年齢","母年齢","血統距離適性","厩舎ランク","調教師","生産者","馬主","蹄","斤量","斤量構成比","馬体重","馬体重増減","重適性","スタート指数","出遅れ率","出遅れ","直線不利","予想IDM","1走前IDM","ベストIDM","中央IDM","1走前着順評価","頭数変化","場所変化","開催変化","負担重量変化","1走前着差","芝ダ変化","最長距離","去勢","距離変化","コース脚質","内_脚質","中_脚質","外_脚質","内_道中","中_道中","外_道中","内_直線","中_直線","外_直線","1走前3コーナー","1走前コーナー順位","テン指数","上がり指数","コース枠番","奇数偶数","調教素点","厩舎素点","CID素点","CID","調教評価","厩舎評価","追切指数","終いF指数","仕上指数","調教パターン","輸送","外厩","放牧先ランク","休養理由","競走馬レベル","間隔","獲得賞金","1走前支持率","ベスト支持率","中央支持率","JRA通算連対率","JRA通算複勝率","キャリア","騎手期待単勝率","騎手期待連対率","騎手期待複勝率","激走指数","LS指数","万券指数","基準支持率"]]
    y_predict = predict[["単勝配当"]]

elif vote == "5" or vote == "6" or vote == "7" or vote == "8" or vote == "9":
    predict = predict_df[(predict_df["投票"]==int(vote))]
    X_predict = predict[["頭数","性別","生後日数","騎手指数","見習い騎手","種牡馬","父系統","母父系統","父年齢","母年齢","血統距離適性","厩舎ランク","調教師","生産者","馬主","蹄","斤量","斤量構成比","馬体重","馬体重増減","重適性","スタート指数","出遅れ率","出遅れ","直線不利","予想IDM","1走前IDM","ベストIDM","中央IDM","1走前着順評価","頭数変化","重量変化","場所変化","開催変化","クラス変化","負担重量変化","1走前着差","芝ダ変化","最長距離","昇級格上","去勢","距離変化","コース脚質","内_脚質","中_脚質","外_脚質","内_道中","中_道中","外_道中","内_直線","中_直線","外_直線","1走前3コーナー","1走前コーナー順位","テン指数","上がり指数","コース枠番","奇数偶数","調教素点","厩舎素点","CID素点","CID","調教評価","厩舎評価","追切指数","終いF指数","仕上指数","調教パターン","輸送","外厩","放牧先ランク","休養理由","競走馬レベル","間隔","獲得賞金","1走前支持率","ベスト支持率","中央支持率","JRA通算勝率","JRA通算連対率","JRA通算複勝率","芝ダ距離別勝率","芝ダ距離別連対率","芝ダ距離別複勝率","トラック別勝率","トラック別連対率","トラック別複勝率","キャリア","騎手期待単勝率","騎手期待連対率","騎手期待複勝率","激走指数","LS指数","万券指数","基準支持率"]]
    y_predict = predict[["単勝配当"]]

elif vote == "10":
    predict = predict_df[(predict_df["投票"]==int(vote))]
    X_predict = predict[["頭数","生後日数","騎手指数","見習い騎手","種牡馬","父系統","母父系統","父年齢","母年齢","血統距離適性","厩舎ランク","調教師","生産者","馬主","蹄","斤量","斤量構成比","馬体重","馬体重増減","重適性","スタート指数","出遅れ率","出遅れ","直線不利","予想IDM","1走前IDM","ベストIDM","中央IDM","1走前着順評価","頭数変化","重量変化","場所変化","開催変化","クラス変化","負担重量変化","1走前着差","芝ダ変化","最長距離","昇級格上","去勢","距離変化","コース脚質","内_脚質","中_脚質","外_脚質","内_道中","中_道中","外_道中","内_直線","中_直線","外_直線","1走前3コーナー","1走前コーナー順位","テン指数","上がり指数","コース枠番","奇数偶数","調教素点","厩舎素点","CID素点","CID","調教評価","厩舎評価","追切指数","終いF指数","仕上指数","調教パターン","輸送","外厩","放牧先ランク","休養理由","競走馬レベル","間隔","獲得賞金","1走前支持率","ベスト支持率","中央支持率","JRA通算勝率","JRA通算連対率","JRA通算複勝率","キャリア","騎手期待単勝率","騎手期待連対率","騎手期待複勝率","激走指数","LS指数","万券指数","基準支持率"]]
    y_predict = predict[["単勝配当"]]

elif vote == "11":
    predict = predict_df[(predict_df["投票"]==int(vote))]
    X_predict = predict[["頭数","性別","生後日数","騎手指数","見習い騎手","種牡馬","父系統","母父系統","父年齢","母年齢","厩舎ランク","調教師","生産者","馬主","蹄","斤量","斤量構成比","馬体重","馬体重増減","1走前着順評価","頭数変化","クラス変化","負担重量変化","昇級格上","去勢","調教素点","厩舎素点","CID素点","CID","調教評価","厩舎評価","追切指数","終いF指数","仕上指数","調教パターン","輸送","外厩","放牧先ランク","休養理由","競走馬レベル","間隔","獲得賞金","1走前支持率","ベスト支持率","中央支持率","JRA通算勝率","JRA通算連対率","JRA通算複勝率","キャリア","騎手期待単勝率","騎手期待連対率","騎手期待複勝率","激走指数","LS指数","万券指数","基準支持率"]]
    y_predict = predict[["単勝配当"]]

st.header("データ分析")
with st.spinner("データ分析中"):
    y_pred = loaded_model.predict(X_predict)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ["pred"]
    y_pred = y_pred.reset_index(drop=True)
    predict = predict.reset_index(drop=True)
    out = pd.concat([predict,y_pred],axis=1)
    prise_thre = st.sidebar.number_input("単勝配当予測閾値",value=100)
    out["購入馬"] = out.apply(lambda x: 1 if (x['pred'] >= prise_thre) else 0, axis=1)
    # 的中馬をマーキング
    out["hitmark"] = out.apply(lambda x: 1 if (x['着順'] == 1 and x['着順'] == x['購入馬']) else 0, axis=1)
    # 日ごと購入馬合計
    by_race = out[["レースキー","日付","hitmark","購入馬","単勝配当"]].groupby(["日付","レースキー"]).sum()
    # 的中率を計算
    acc_rate = out['hitmark'].sum() / len(by_race[by_race["購入馬"]>0]) * 100
    # 回収率を計算
    pay_back = out.loc[out['hitmark'] == 1, '単勝配当'].sum() / (out.loc[out['購入馬'] == 1, '購入馬'].sum()*100) * 100
    by_race["レース毎配当"] = by_race.apply(lambda x: x["単勝配当"] if (x['hitmark'] == 1 ) else 0, axis=1)
    by_race["レース毎損益"] = by_race.apply(lambda x: x["単勝配当"] - x["購入馬"]*100 if (x['hitmark'] == 1 ) else - x["購入馬"]*100, axis=1)
    by_race["累積損益"] = by_race["レース毎損益"].cumsum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("レース数",str(len(out['レースキー'].unique()))+"レース")
        st.metric("購入馬券数",str(len(out[out["購入馬"]>0]))+"枚")
        st.metric("的中馬券数",str(out['hitmark'].sum())+"枚")
    with col2:
        st.metric("的中率", str(int(acc_rate)) + "%")
        st.metric("回収率", str(int(pay_back)) + "%")
        st.metric("回収額", "¥"+str(by_race["累積損益"].tail(1).iloc[0]))
    
    by_race_data = by_race.reset_index()
    by_race_data["日付"] = pd.to_datetime(by_race_data['日付'], format='%Y%m%d')
    day= list(by_race_data["日付"].apply(str).str[:10])

    by_date = by_race_data.groupby(pd.Grouper(key="日付", freq="M")).sum()
    by_date = by_date.reset_index()
    day2= list(by_date["日付"].apply(str).str[:10])
   
    v1 = by_race_data["累積損益"]
    c = line_chart(day,v1,"累積損益")
    d = bar_chart(day2,by_date["購入馬"].values.tolist(),"購入馬券数")
    e = bar_chart(day2,by_date["hitmark"].values.tolist(),"的中馬券数")
    month_ticket = e.overlap(d)

    month_payback = (by_date["レース毎損益"] /(by_date["購入馬"]*100))*100
    month_payback_chart = bar_chart(day2,month_payback.values.tolist(),"月別回収率")
    st.subheader("累積損益")
    st_pyecharts(c)

    st.subheader("月別馬券数")
    st_pyecharts(month_ticket)
    st.subheader("月別回収率")
    st_pyecharts(month_payback_chart)