import numpy as np
import cv2

video = cv2.VideoCapture('test_vids\\1.mp4')
frame_width = int(video.get(3))
frame_height = int(video.get(4))

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = np.ones((5,5),np.uint8)
out = cv2.VideoWriter("test_vids\\outpy.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(1):
    ret, orig_frame = video.read()
    if not ret:
        break

    frame = fgbg.apply(orig_frame)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    
    edges = cv2.Canny(frame, 50, 150)
    # cv2.imshow("frame",edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=100)
    
    a1=0
    a2=0
    b1=-10000000
    b2=100000000
    count = 0
    if lines is not None:
        # print(lines[0][0].size)
        x_list1 = list()
        x_list2 = list()
        
        # print(lines.size)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 > y2:
                x_list1.append(x1)
                b1 = max(b1,y1)
                x_list2.append(x2)
                b2 = min(b2,y2)
            else:
                x_list1.append(x2)
                b1 = max(b1,y2)
                x_list2.append(x1)
                b2 = min(b2,y1)
            # cv2.line(orig_frame, (x1,y1), (x2,y2), (128,0,128), 2)
            count+=1
        a1 = x_list1[len(x_list1)//2] 
        a2 = x_list2[len(x_list2)//2] 
        
        cv2.line(orig_frame, (a1,b1), (a2,b2), (128,0,128), 2)
    
    cv2.imshow("frame", orig_frame)
    out.write(orig_frame)
    
    # cv2.imshow("edges", edges)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
out.release()
cv2.destroyAllWindows()
    
