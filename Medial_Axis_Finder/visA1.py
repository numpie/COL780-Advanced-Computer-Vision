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
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    
    edges = cv2.Canny(frame, 100, 200)
    # cv2.imshow("frame",edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=100)
    
    a1=0
    a2=0
    b1=-10000000
    b2=100000000
    count = 0
    cleaned_lines = list()
    x_list1 = list()
    slope_list = list()
    if lines is not None:
        # print(lines[0][0].size)
        
        # y_list1 = list()
        # y_list2 = list()
        # for 
        # print(lines.size)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            m = (y2-y1)/(x2-x1)
            slope_list.append((m,line[0]))
            # cv2.line(frame, (a1,b1), (a2,0), (0,0,255), 2)

        slopes = [x[0] for x in slope_list]
        sigma = np.std(slopes)
        meu = np.mean(slopes)

        cleaned_lines = [x[1] for x in slope_list if abs(x[0]-meu)<1.5*sigma]
    
    if len(cleaned_lines) > 0:
        print()
        for line in cleaned_lines:
            x1, y1, x2, y2 = line
            if y1 > y2:
                x_list1.append((x1,x2,y1))
                # y_list1.append(y1)
                # y_list2.append(y2)
                # b1 = max(b1,y1)
                # b2 = min(b2,y2)
            else:
                x_list1.append((x2,x1,y2))
                # y_list1.append(y2)
                # y_list2.append(y1)
                # b1 = max(b1,y2)
                # b2 = min(b2,y1)
            # cv2.line(orig_frame, (x1,y1), (x2,y2), (255,0,0), 2)
        x_list1.sort()
        # y_list1.sort()
        # y_list2.sort()
        print(x_list1)
        a1,a2,b1 = x_list1[len(x_list1)//2] 
        
        # b2 = y_list1[len(y_list1)//2]
        # b2 = y_list2[len(y_list2)//2]
        cv2.line(orig_frame, (a1,b1), (a2,0), (128,0,128), 2)
    
    cv2.imshow("frame", orig_frame)
    # cv2.imshow("frame2",frame)
    out.write(orig_frame)
    
    # cv2.imshow("edges", edges)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
out.release()
cv2.destroyAllWindows()
    
