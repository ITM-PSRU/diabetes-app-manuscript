import streamlit as st
import numpy as np
import pandas as pd
import  os
from PIL import Image
import pickle
import sys
from pip import __main__


with open('model.pkl','rb') as model:
    model_dt = pickle.load(model)

def main():
    st.set_page_config(page_title="Diabetes Health Check App", page_icon="⚕️", layout="centered",initial_sidebar_state="auto")

    html_temp = """ 
        <div style ="background-color:pink;padding:11px"> 
        <h1 style ="color:black;text-align:center;">Prediction Diabetes</h1> 
        </div><br/>
         <h5 style ="text-align:center;">เว็บแอปพลิเคชันทำนายความเสี่ยงเป็นโรคเบาหวาน</h5>
         กรุณากรอกข้อมูลด้านล่างนี้ ให้ครบถ้วน
        """

    html_sidebar = """
        <div style ="background-color:pink;padding:10px"> 
        <h1 style ="color:black;text-align:center;">สาระสุขภาพ</h1>
                <p style ="color:black;">เนื้อหาข้อมูลด้านล่างนี้  มาจากเว็บไซต์ของ โรงพยาบาลศิริราช ปิยมหาราชการุณย์ ที่ให้ความรู้และให้คำปรึกษาด้านสุขภาพต่างๆ ที่เป็นประโยชน์ที่ดีต่อท่าน</p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-2" target="_blank">เบาหวาน รู้ทันป้องกันได้</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetic-diet" target="_blank">เบาหวาน ควรทานอย่างไร</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-exercise" target="_blank">ออกกำลังกายพิชิตเบาหวาน</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-mellitus" target="_blank">ภาวะแทรกซ้อนจากโรคเบาหวาน</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetic-retinopathy" target="_blank">เบาหวานขึ้นจอตา อันตรายแค่ไหน</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-guides" target="_blank">เมื่อเจ็บป่วยควรทำอย่างไร</a></p> 
    </div> 
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.write(html_sidebar, unsafe_allow_html=True)


    #st.sidebar.subheader("About App")
    #st.sidebar.info('คำนวณค่าดัชนีมวลกาย BMI')
    #weight = st.sidebar.number_input(label="ป้อนน้ำหนักของคุณ (kg)", min_value=0, max_value=None, value=1, step=1)
    #high = st.sidebar.number_input(label="ป้อนส่วนสูงของคุณ (cm)", min_value=0, max_value=None, value=1,step=1) / 100
    #bmi = weight / (high ** 2)
    #num = int(bmi)
    #st.sidebar.subheader("ค่า BMI ของคุณ คือ : ")
    #st.sidebar.info(num)

    Sex = st.selectbox("เพศ", ("หญิง", "ชาย"))
    if Sex=='หญิง':
        Sex=0
    elif Sex=='ชาย':
        Sex=1

    Age = st.selectbox("ช่วงระดับ อายุของคุณ", ("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"))
    if Age=='18-24':
        Age=1
    elif Age=='25-29':
        Age=2
    elif Age=='30-34':
        Age=3
    elif Age=='35-39':
        Age=4
    elif Age=='40-44':
        Age=5
    elif Age=='45-49':
        Age=6
    elif Age=='50-54':
        Age=7
    elif Age=='55-59':
        Age=8
    elif Age=='60-64':
        Age=9
    elif Age=='65-69':
        Age=10
    elif Age=='70-74':
        Age=11
    elif Age=='75-79':
        Age=12
    elif Age =='80+':
        Age=13

    weight = st.number_input(label="ระบุน้ำหนักของคุณ (kg)", min_value=0, max_value=None, value=1, step=1)
    high = st.number_input(label="ระบุส่วนสูงของคุณ (cm)", min_value=0, max_value=None, value=1,step=1) / 100
    # bmi = weight / (high ** 2)
    BMI = weight / (high ** 2)

    HighBP = st.radio("ระดับความดันโลหิตของคุณ สูงมากกว่า 120/80 ใช่หรือไม่?", ["ไม่ใช่", "ใช่"])
    if HighBP=='ไม่ใช่':
        HighBP= 0
    elif HighBP=='ใช่':
        HighBP=1

    HighBS = st.radio("ระดับน้ำตาลในเลือดของคุณ สูงมากกว่า 100 - 125 มิลลิกรัมต่อเดซิลิตร ใช่หรือไม่?", ["ไม่ใช่", "ใช่"])
    if HighBS=='ไม่ใช่':
        HighBS=0
    elif HighBS=='ใช่':
        HighBS=1

    #CholCheck = st.radio("คุณได้รับการตรวจคอลเลสเตอรอล ภายในรอบ 5 ปี ใช่หรือไม่?",["ไม่ใช่", "ใช่"])
    #if CholCheck=='ไม่ใช่':
        #CholCheck=0
    #elif CholCheck=='ใช่':
        #CholCheck=1

    Smoker = st.radio("คุณสูบบุหรี่มาตลอด มากกว่า 5 ซอง ใช่หรือไม่?",["ไม่ใช่", "ใช่"])
    if Smoker=='ไม่ใช่':
        Smoker=0
    elif Smoker=='ใช่':
        Smoker=1

    #Stroke = st.radio("คุณเคยมี โรคหลอดเลือดสมอง ใช่หรือไม่?", ["ไม่ใช่", "ใช่"])
    #if Stroke=='ไม่ใช่':
        #Stroke=0
    #elif Stroke=='ใช่':
        #Stroke=1

    #HeartDiseaseorAttack = st.radio("คุณมีโรคหลอดเลือดหัวใจ หรือกล้ามเนื้อหัวใจตาย ใช่หรือไม่?",["ไม่ใช่", "ใช่"])
    #if HeartDiseaseorAttack == 'ไม่ใช่':
       #HeartDiseaseorAttack = 0
    #elif HeartDiseaseorAttack == 'ใช่':
        #HeartDiseaseorAttack = 1

    PhysActivity = st.radio("ช่วง 30 วันที่ผ่านมาคุณได้ออกกำลัง บ้างหรือไม่?", ["ไม่ใช่", "ใช่"])
    if PhysActivity=='ไม่ใช่':
        PhysActivity=0
    elif PhysActivity=='ใช่':
        PhysActivity=1

    Fruits = st.radio("ใน 1 วัน คุณทานผลไม้ บ้างหรือไม่?", ["ไม่ใช่", "ใช่"])
    if Fruits=='ไม่ใช่':
        Fruits=0
    elif Fruits=='ใช่':
        Fruits=1

    Veggies = st.radio("ใน 1 วัน คุณทานผัก บ้างหรือไม่?", ["ไม่ใช่", "ใช่"])
    if Veggies=='ไม่ใช่':
        Veggies=0
    elif Veggies=='ใช่':
        Veggies=1

    HvyAlcoholConsump = st.radio("ปริมาณการดื่มแอลกอฮอล์ ต่อสัปดาห์ [ผู้ชายมากกว่า 13 แก้ว] , [ผู้หญิงมากกว่า 6 แก้ว]", ["ไม่ใช่", "ใช่"])
    if HvyAlcoholConsump=='ไม่ใช่':
        HvyAlcoholConsump=0
    elif HvyAlcoholConsump=='ใช่':
        HvyAlcoholConsump=1

    #AnyHealthcare = st.radio("คุณมีความคุ้มครองหรือประกันสุขภาพ หรือไม่?",["ไม่มี", "มี"])
    #if AnyHealthcare=='ไม่มี':
        #AnyHealthcare=0
    #elif AnyHealthcare=='มี':
        #AnyHealthcare=1

    #NoDocbcCost = st.radio("ช่วง 12 เดือนที่ผ่านมา คุณมีความต้องการพบแพทย์ แต่ไม่ได้เข้าพบ เพราะต้องเสียค่าใช้จ่าย บ้างหรือไม่?",["ไม่ใช่", "ใช่"])
    #if NoDocbcCost=='ไม่ใช่':
        #NoDocbcCost=0
    #elif NoDocbcCost=='ใช่':
        #NoDocbcCost=1

    #MentHlth = st.slider("วันที่คุณมีความเครียดรวมถึงภาวะซึมเศร้า และปัญหาเกี่ยวกับอารมณ์ คุณเป็นกี่วัน ในช่วง 30 วันที่ผ่านมา",min_value=0, max_value=30, value=0, step=1)

    #PhysHlth = st.slider("คุณมีความเจ็บป่วยทางร่างกายหรือบาดเจ็บ เป็นเวลากี่วัน ในช่วง 30 วันที่ผ่านมา", min_value=0,max_value=30, value=0, step=1)

    DiffWalk = st.radio("คุณมีปัญหาร้ายแรงในการเดิน หรือขึ้นลงบันได หรือไม่", ["ไม่มี", "มี"])
    if DiffWalk=='ไม่มี':
        DiffWalk=0
    elif DiffWalk=='มี':
        DiffWalk=1

    Education = st.selectbox("ระดับการศึกษาของคุณ", ("ไม่เคยเข้าโรงเรียนหรือเรียนอนุบาลเท่านั้น", "ประถมศึกษา", "มัธยมศึกษาตอนต้น(ม.ต้น)",
        "มัธยมศึกษาตอนปลายหรือโรงเรียนเทคนิค(ม.ปลาย,ปวช.)", "วิทยาลัยหรือโรงเรียนเทคนิคบางแห่ง(ปวส.)","บัณฑิตวิทยาลัย 4 ปีขึ้นไป"))
    if Education=='ไม่เคยเข้าโรงเรียนหรือเรียนอนุบาลเท่านั้น':
        Education=1
    elif Education=='ประถมศึกษา':
        Education=2
    elif Education=='มัธยมศึกษาตอนต้น(ม.ต้น)':
        Education=3
    elif Education=='มัธยมศึกษาตอนปลายหรือโรงเรียนเทคนิค(ม.ปลาย,ปวช.)':
        Education=4
    elif Education=='วิทยาลัยหรือโรงเรียนเทคนิคบางแห่ง(ปวส.)':
        Education=5
    elif Education=='บัณฑิตวิทยาลัย 4 ปีขึ้นไป':
        Education=6

    GenHlth = st.selectbox("คุณคิดว่าสุขภาพของคุณตอนนี้เป็นอย่างไร?", ("สุขภาพแข็งแรงดีมาก", "สุขภาพดี", "สุขภาพทรงตัว เริ่มรู้สึกว่าร่างกายอ่อนแอลง", "สุขภาพแย่ เจ็บป่วยบ่อยครั้ง", "แย่มาก ต้องเข้ารับการรักษาที่โรงพยาบาลบ่อย"))
    if GenHlth=='สุขภาพแข็งแรงดีมาก':
        GenHlth=1
    elif GenHlth=='สุขภาพดี':
        GenHlth=2
    elif GenHlth=='สุขภาพทรงตัว เริ่มรู้สึกว่าร่างกายอ่อนแอลง':
        GenHlth=3
    elif GenHlth=='สุขภาพแย่ เจ็บป่วยบ่อยครั้ง':
        GenHlth=4
    elif GenHlth == 'แย่มาก ต้องเข้ารับการรักษาที่โรงพยาบาลบ่อย':
        GenHlth=5

    #Income = st.selectbox("รายได้เฉลี่ย ต่อเดือนของคุณ", ("น้อยกว่า 30,000 บาท", "น้อยกว่า 45,000 บาท", "น้อยกว่า 60,000 บาท", "น้อยกว่า 75,000 บาท","น้อยกว่า 105,000 บาท","น้อยกว่า 150,000 บาท", "น้อยกว่า 225,000 บาท", "มากกว่า 225,000+ บาท",))
    #if Income=='น้อยกว่า 30,000 บาท':
        #Income=1
    #elif Income=='น้อยกว่า 45,000 บาท':
        #Income=2
    #elif Income=='น้อยกว่า 60,000 บาท':
        #Income=3
    #elif Income=='น้อยกว่า 75,000 บาท':
        #Income=4
    #elif Income=='น้อยกว่า 105,000 บาท':
        #Income=5
    #elif Income=='น้อยกว่า 150,000 บาท':
        #Income=6
    #elif Income=='น้อยกว่า 225,000 บาท':
        #Income=7
    #elif Income=='มากกว่า 225,000+ บาท':
        #Income=8
    if st.button('ทำนายผล'):
        result = prediction (HighBP,HighBS,BMI,Smoker,PhysActivity,Fruits,
                            Veggies,HvyAlcoholConsump,GenHlth,DiffWalk,Sex,Age,Education)
        if (result == 1):
            st.warning('คำเตือน! คุณมีโอกาสเสี่ยงสูงที่จะเป็นโรคเบาหวาน')
        elif (result == 0):
            st.success('คุณมีสุขภาพดีและมีโอกาสเสี่ยงน้อยที่จะเป็นโรคเบาหวาน')


def prediction(HighBP, HighBS,BMI, Smoker, Stroke, Fruits, Veggies,HvyAlcoholConsump,GenHlth,DiffWalk, Sex, Age,Education,):
    predicted_output = model_dt.predict([[HighBP, HighBS,BMI, Smoker, Stroke, Fruits, Veggies,HvyAlcoholConsump, GenHlth, DiffWalk, Sex, Age,Education,]])
    return predicted_output



if __name__ == '__main__':
    main()







