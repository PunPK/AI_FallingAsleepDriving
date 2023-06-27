# AI_FallingAsleepDriving

   เนื่องจากในปัจจุบัน มีอุบัติเหตุที่เกิดมาจาก “การหลับใน” เพิ่มมากขึ้น จึงทำการแก้ปัญหาใช้ AI มาตรวจจับ โดยการสร้าง “AI ตรวจสอบผู้ขับขี่หลับใน” หรือ “AI FallingAsleepDriving” สร้างแบบใช้ image classification + object detection ซึ่งเป็น AI ที่มีกระบวนการ object detection จากตาและปาก แล้วมา image classification วิเคราะห์การเปิด/ปิดของดวงตา และวิเคราะห์การเปิด/ปิดของปาก ขณะกำลังขับรถยนต์หรือพาหนะอื่นๆ เครื่องมือจะทำการแจ้งเตือนเมื่อตรวจสอบแล้วพบว่า มีการหลับตา/ปิดตามากกว่าปกติ หรือมีการเปิดปากมาก/หาวมากกว่าปกติ ในขณะช่วงเวลาหนึ่ง โดย Project นี้เป็นการดำเนินงานเฉพาะในส่วนของ image classification ทำการเทรนโมเดลผ่าน Colab โดยแบ่ง Data เทรนออกเป็น 2 class คือ “ตา” กับ “ปาก” และใช้ backbone 2 ชนิด มาเปรียบเทียบกัน คือ resnet18 กับ C﻿NN แบบกำหนดเอง  ได้ค่า Accuracy 100% และ 95% ตามลำดับ แล้วนำโมเดลมารวมกับ object detection มาใช้ใน VScode โดยใช้งานกล้องจาก Webcam สามารถนับจำนวนปิดตา/กระพริบตา/หาวได้ปกติ และสามารถแสดงค่าข้อมูลบนหน้าจอรวมทั้งมีการเตือนรูปแบบเสียง/ข้อความเมื่อตรวจสอบพบการหลับในขณะขับขี่ ต่อมาได้ Deployment ออกมาเป็น application.exe กับบน huggingface ซึ่งยังสามารถใช้ได้แต่ยังมีระยะเวลาในการประมวลผลนาน ทำให้ค่า FPS ค่อนข้างน้อย แต่ก็สามารถนำมาใช้ดูการ หลับตา/หาว โดยเบื้องต้นได้ และต้องใช้ในที่ๆมีแสงสว่างส่องบริเวณใบหน้าที่เหมาะสม

จากโครงการ Ai builders ปีที่ 3 ประจำปี 2023

Github : https://github.com/PunPK/AI_FallingAsleepDriving

Medium : https://medium.com/@phurutsakorn.kps/️-ai-fallingasleepdriving-993ce04ccfcc

Huggingface spaces : https://huggingface.co/spaces/PunPk/AI_FallingAsleepDriving

คลิปทดลองใช้จริง : https://www.youtube.com/watch?v=mi2gX3Wmeks

ทั้งนี้ผมต้องขอขอบคุณ พี่ TA และ Mentor ของกลุ่ม delta-ducks ด้วยครับ รวมทั้งโครงการ Ai builders ปีที่ 3 ประจำปี 2023 ที่เป็นโครงการที่ทำให้ผมได้เริ่มเรียนรู้การสร้างโมเดล AI ตั้งแต่เริ่มเก็บ Data ไปจนถึง Deployment ครับ ถ้าไม่มีพี่ TA และ Mentor หรือ โครงการ Ai builders ที่มาช่วยสอนหรือแนะนำวิธีที่ถูกต้องให้ผมในการสร้างโมเดล AI ผมคงไม่ได้ทำ Ai เสร็จจนถึง Deployment ได้เร็วขนาดนี้ครับ ต้องขอบคุณมากๆจริงๆครับ

![image](https://github.com/PunPK/AI_FallingAsleepDriving/assets/129741543/ea76c28a-71f3-4c10-a0fc-c930ed8eda17)
