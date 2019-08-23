import numpy as np
import cv2
from sklearn.cluster import KMeans

for vid_num in range(1,11):
    video = cv2.VideoCapture('test_vids\\'+str(vid_num)+'.mp4')
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    kmeans = KMeans(n_clusters=2) # K mean cluster

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() #background subtractor
    kernel = np.ones((5,5),np.uint8)
    out = cv2.VideoWriter("test_vids\\out"+str(vid_num)+".avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    axis_X_point1=0 
    axis_X_point2=0
    prev_medial_axis = [0,0,0]
    queue_size = 5
    slope_queue = list()
    no_line_detect_count = 0

    while(1):
        ret, orig_frame = video.read()

        #Stop when no frame found
        if not ret:
            break

        # Apply background subtractor on the frame
        frame = fgbg.apply(orig_frame) 
        
        # Apply erosion followed by dialation to clean the image
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        
        # Blurring the image
        frame = cv2.GaussianBlur(frame, (3,3), 0)
        
        # Detect edges using Canny edge detector
        edges = cv2.Canny(frame, 100, 200)

        # Probabilistic Hough Transform to detect lines from the edges
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=75)
        
        axis_Y_point1=-10000000
        cleaned_lines = list()
        x_intercept_and_slope_list = list()
        slope_list = list()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2!=x1 :
                    m = (y2-y1)/(x2-x1)
                    slope_list.append((m,line[0]))

            slopes = [x[0] for x in slope_list]

            sigma = np.std(slopes) # Standard Deviation of all the slopes
            meu = np.mean(slopes) # Mean of all the slopes

            # Removing the lines from the list that doesn't come within the range of mean+/-1.5*sigma
            cleaned_lines = [x for x in slope_list if abs(x[0]-meu)<1.5*sigma]
        
        if len(cleaned_lines) > 0:
            no_line_detect_count = 0

            for line in cleaned_lines:
                x1, y1, x2, y2 = line[1]
                m = line[0]
                if y1 > y2:
                    x2 = -(y2/m) + x2
                    x_intercept_and_slope_list.append((x2, m))
                    axis_Y_point1 = max(axis_Y_point1,y1)
                else:
                    x1 = -(y1/m) + x1
                    x_intercept_and_slope_list.append((x1, m))
                    axis_Y_point1 = max(axis_Y_point1, y2)

            x_intercept_list = [[x[0],] for x in x_intercept_and_slope_list]

            # Clustering lines in two parts
            if len(x_intercept_list)>1:
                clusters = kmeans.fit(x_intercept_list)
                centers = kmeans.cluster_centers_
                axis_X_point2 = int((centers[0][0] + centers[1][0])/2)

            m = x_intercept_and_slope_list[len(x_intercept_and_slope_list)//2][1]
            slope_queue.append(m)

            if len(slope_queue) > queue_size:
                slope_queue.pop(0)

            # Taking median of all the slopes stored previoulsy and the current one
            m = sorted(slope_queue)[len(slope_queue)//2]
            axis_X_point1 = int((axis_Y_point1/m)+ axis_X_point2+ 0.5)
            cv2.line(orig_frame, (axis_X_point1, axis_Y_point1), (axis_X_point2, 0), (0, 0, 255), 2)
            prev_medial_axis = [axis_X_point1, axis_X_point2, axis_Y_point1]

        # Displaying the previous medial axis when no lines detected
        else:
            axis_X_point1, axis_X_point2, axis_Y_point1 = prev_medial_axis
            no_line_detect_count +=1

            if no_line_detect_count < 20:
                cv2.line(orig_frame, (axis_X_point1, axis_Y_point1), (axis_X_point2, 0), (0, 0, 255), 2)

        out.write(orig_frame)
        
    video.release()
    out.release()
cv2.destroyAllWindows()