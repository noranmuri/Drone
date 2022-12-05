import cv2

cap = cv2.VideoCapture("drone-ycenter.MOV")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, 30.0, ((int)(width), (int)(height)))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break
    
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
