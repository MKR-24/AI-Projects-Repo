import cv2
import face_recognition


def load_and_prepare_image(image_path):
    try:
        img = face_recognition.load_image_file(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


img1_path = 'Images/Image1.png'
imgTest_path = 'Images/Image2.png'
img1 = load_and_prepare_image(img1_path)
imgTest = load_and_prepare_image(imgTest_path)

if img1 is not None and imgTest is not None:
    try:
       
        faceLoc1 = face_recognition.face_locations(img1)
        if faceLoc1:
            faceLoc1 = faceLoc1[0]
            encodeImg1 = face_recognition.face_encodings(img1)[0]
            cv2.rectangle(img1, (faceLoc1[3], faceLoc1[0]), (faceLoc1[1], faceLoc1[2]), (255, 0, 255), 2)
        else:
            print("No face detected in the first image.")
            encodeImg1 = None

        
        faceLocTest = face_recognition.face_locations(imgTest)
        if faceLocTest:
            faceLocTest = faceLocTest[0]
            encodeImgTest = face_recognition.face_encodings(imgTest)[0]
            cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
        else:
            print("No face detected in the test image.")
            encodeImgTest = None

        
        if encodeImg1 is not None and encodeImgTest is not None:
            results = face_recognition.compare_faces([encodeImg1], encodeImgTest)
            faceDis = face_recognition.face_distance([encodeImg1], encodeImgTest)
            print(f"Match: {results}, Distance: {faceDis}")

           
            cv2.putText(imgTest, f'Match: {results[0]} Distance: {round(faceDis[0], 2)}', 
                        (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        print(f"An error occurred: {e}")

    
    cv2.imshow('Image 1', img1)
    cv2.imshow('Image 2', imgTest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Images could not be loaded. Please check the file paths.")
