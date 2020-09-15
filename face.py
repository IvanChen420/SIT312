import os
image_list = []

path = "/Users/ivan/Downloads/lfw/"

files = os.listdir(path)
for root, directory, f in os.walk(path):
    for name in f:
        if name.endswith(".jpg"):
            image_list.append(os.path.join(root,name))

#image_list.append("/Users/ivan/Downloads/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
import face_recognition
import cv2
import numpy as np
video_capture = cv2.VideoCapture(0)
known_face_encodings = []
for i in image_list:
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(i)))

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True    

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Counter"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)


            face_names.append(name)

    process_this_frame = not process_this_frame
    
    def track():
    image1 = vs.read()   # initialize image1 (done once)
    grayimage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    cx, cy, cw, ch = 0, 0, 0, 0   # initialize contour center variables
    still_scanning = True
    movelist = []
    move_time = time.time()
    enter = 0
    leave = 0
    while still_scanning:
        # initialize variable
        motion_found = False
        biggest_area = MIN_AREA
        image2 = vs.read()  # initialize image2
        if WINDOW_ON:
            if CENTER_LINE_VERT:
                cv2.line(image2, (X_CENTER, 0), (X_CENTER, Y_MAX), COLOR_TEXT, 2)
            else:
                cv2.line(image2, (0, Y_CENTER), (X_MAX, Y_CENTER), COLOR_TEXT, 2)
        grayimage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # Get differences between the two greyed images
        difference_image = cv2.absdiff(grayimage1, grayimage2)
        # save grayimage2 to grayimage1 ready for next image2
        grayimage1 = grayimage2
        difference_image = cv2.blur(difference_image, (BLUR_SIZE, BLUR_SIZE))
        # Get threshold of difference image based on
        # THRESHOLD_SENSITIVITY variable
        retval, thresholdimage = cv2.threshold(difference_image,
                                               THRESHOLD_SENSITIVITY, 255,
                                               cv2.THRESH_BINARY)
        # Try python2 opencv syntax and fail over to
        # python3 opencv syntax if required
        #try:
        contours, hierarchy = cv2.findContours(thresholdimage,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        #except ValueError:
        #    thresholdimage, contours, hierarchy = cv2.findContours(thresholdimage,
        #                                                           cv2.RETR_EXTERNAL,
        #                                                           cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            total_contours = len(contours)  # Get total number of contours
            for c in contours:              # find contour with biggest area
                found_area = cv2.contourArea(c)  # get area of next contour
                # find the middle of largest bounding rectangle
                if found_area > biggest_area:
                    motion_found = True
                    biggest_area = found_area
                    (x, y, w, h) = cv2.boundingRect(c)
                    cx = int(x + w/2)   # put circle in middle of width
                    cy = int(y + h/2)   # put circle in middle of height
                    cw, ch = w, h
            if motion_found:
                move_timer = time.time() - move_time
                if move_timer >= MOVE_LIST_TIMEOUT:
                    movelist = []
                move_time = time.time()
                old_enter = enter
                old_leave = leave
                if CENTER_LINE_VERT:
                    movelist.append(cx)
                    enter, leave, movelist = crossed_x_centerline(enter, leave, movelist)
                else:
                    movelist.append(cy)
                    enter, leave, movelist = crossed_y_centerline(enter, leave, movelist)
                if not movelist:
                    if enter > old_enter:
                        if INOUT_REVERSE:   # reverse enter leave if required
                            prefix = "leave"
                        else:
                            prefix = "enter"
                    elif leave > old_leave:
                        if INOUT_REVERSE:
                            prefix = enter
                        else:
                            prefix = "leave"
                    else:
                        prefix = "error"
                if WINDOW_ON:
                    # show small circle at motion location
                    if SHOW_CIRCLE and motion_found:
                        cv2.circle(image2, (cx, cy), CIRCLE_SIZE,
                                   COLOR_MO, LINE_THICKNESS)
                    else:
                        cv2.rectangle(image2, (cx, cy), (x+cw, y+ch),
                                      COLOR_MO, LINE_THICKNESS)
        if WINDOW_ON:
            if INOUT_REVERSE:
                img_text = ("LEAVE %i          ENTER %i" % (leave, enter))
            else:
                img_text = ("ENTER %i          LEAVE %i" % (enter, leave))
            cv2.putText(image2, img_text, (35, 15),
                        TEXT_FONT, FONT_SCALE, (COLOR_TEXT), 1)
            cv2.imshow('Window',image2)
            if DIFF_WINDOW_ON:
                cv2.imshow('Difference Image', difference_image)
            if THRESH_WINDOW_ON:
                cv2.imshow('OpenCV Threshold', thresholdimage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                vs.stop()
                print("End Motion Tracking")
                quit(0)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
