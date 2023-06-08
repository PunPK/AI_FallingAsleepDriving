import cv2

# เปิดวิดีโอ
video = cv2.VideoCapture("test.mp4")

# ตั้งค่าขนาดวิดีโอที่ต้องการอ่าน (ให้เป็น 360p)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# ตั้งค่า FPS ที่ต้องการอ่าน (ให้เป็น 30 fps)
video.set(cv2.CAP_PROP_FPS, 30)

# อ่านวิดีโอเฟรมต่อเนื่องจนกว่าจะสิ้นสุด
while True:
    ret, frame = video.read()

    if not ret:
        break

    # ทำสิ่งที่ต้องการกับเฟรมที่อ่านได้ที่นี่
    # เช่นการประมวลผลวิดีโอ หรือแสดงผลเฟรม

    cv2.imshow("Video", frame)

    # หยุดการแสดงผลเมื่อกด 'q' บนแป้นพิมพ์
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# คืนทรัพยากร
video.release()
cv2.destroyAllWindows()
