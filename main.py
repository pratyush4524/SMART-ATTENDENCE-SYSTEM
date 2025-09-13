import os
import cv2
import pickle
import face_recognition
import numpy as np

# Load the encoding file
print("Loading encode files...")
file = open("encoding.p", "rb")
encodeListKnownwithides = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownwithides
print("Encode file loaded")

# Configuration
confidenceThreshold = 0.6  # Adjust this value (lower = more strict)

def detect_faces_in_image(image_path):
    """Detect faces in a single image"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Convert to RGB for face_recognition
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    
    print(f"Found {len(face_locations)} face(s) in the image")
    
    # Process each detected face
    detected_faces = []
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding, tolerance=confidenceThreshold)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < confidenceThreshold:
                # Known face
                student_id = studentIds[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                detected_faces.append({
                    'id': student_id,
                    'location': face_location,
                    'confidence': confidence,
                    'known': True
                })
            else:
                # Unknown face
                detected_faces.append({
                    'id': 'Unknown',
                    'location': face_location,
                    'confidence': 1 - min(face_distances) if len(face_distances) > 0 else 0,
                    'known': False
                })
        else:
            # No known faces in database
            detected_faces.append({
                'id': 'Unknown',
                'location': face_location,
                'confidence': 0,
                'known': False
            })
    
    # Draw bounding boxes and labels
    for face in detected_faces:
        top, right, bottom, left = face['location']
        
        # Choose color based on recognition
        color = (0, 255, 0) if face['known'] else (0, 0, 255)  # Green for known, Red for unknown
        
        # Draw rectangle around face
        cv2.rectangle(img, (left, top), (right, bottom), color, 3)
        
        # Add label
        label = f"{face['id']} ({face['confidence']:.2f})"
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Add summary text
    summary = f"Total Faces: {len(detected_faces)} | Known: {sum(1 for f in detected_faces if f['known'])} | Unknown: {sum(1 for f in detected_faces if not f['known'])}"
    cv2.putText(img, summary, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Print results
    print("\nDetection Results:")
    for i, face in enumerate(detected_faces, 1):
        status = "Known" if face['known'] else "Unknown"
        print(f"{i}. {face['id']} - {status} (Confidence: {face['confidence']:.2f})")
    
    return img, detected_faces

def main():
    # Get image path from user
    image_path = "resources/pig.png"
    
    # Remove quotes if user added them
    image_path = image_path.strip('"').strip("'")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return
    
    # Process the image
    result_img, faces = detect_faces_in_image(image_path)
    
    if result_img is not None:
        # Display the result
        cv2.namedWindow("Face Detection Results", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Face Detection Results", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Face Detection Results", result_img)
        
        print(f"\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optionally save the result
        save_result = input("\nDo you want to save the result image? (y/n): ").strip().lower()
        if save_result == 'y':
            output_path = f"face_detection_result_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, result_img)
            print(f"Result saved as: {output_path}")

if __name__ == "__main__":
    main()