import cv2
import numpy as np
import dlib

# Paths to the resources
overlay_image_path = 'cristiano_ronaldo.png'
landmark_dat_path = 'shape_predictor_68_face_landmarks.dat'  # Ensure this file is in the same directory or provide a full path

# Load the overlay image
overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

# Initialize dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(landmark_dat_path)

# Function to create a mask from landmark points
def create_mask_from_landmarks(image_shape, points):
    mask = np.zeros(image_shape[:2], dtype=np.float32)  # create a 2D mask
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask

# Function to warp overlay to match the face landmarks
def warp_overlay(overlay, points, dest_points):
    overlay_height, overlay_width = overlay.shape[:2]
    M, _ = cv2.findHomography(points, dest_points)
    warped_overlay = cv2.warpPerspective(overlay, M, (overlay_width, overlay_height), None,
                                         cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
    return warped_overlay

# Start the video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(grayscale_frame)

    for face in faces:
        landmarks = shape_predictor(grayscale_frame, face)
        landmarks_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], dtype=np.float32)

        # Select points corresponding to the eye corners and mouth corners
        src_points = np.array([
            landmarks_points[36],  # Left eye left corner
            landmarks_points[45],  # Right eye right corner
            landmarks_points[48],  # Left mouth corner
            landmarks_points[54]   # Right mouth corner
        ])
        
        # Define destination points in the overlay
        overlay_points = np.array([
            [0, 0],
            [overlay_image.shape[1] - 1, 0],
            [0, overlay_image.shape[0] - 1],
            [overlay_image.shape[1] - 1, overlay_image.shape[0] - 1]
        ], dtype=np.float32)

        # Warp the overlay image to fit the detected face landmarks
        warped_overlay = warp_overlay(overlay_image, overlay_points, src_points)

        # Extract the alpha channel and create masks
        if warped_overlay.shape[2] == 4:
            alpha_channel = warped_overlay[:, :, 3] / 255.0
            overlay_rgb = warped_overlay[:, :, :3]

            mask = create_mask_from_landmarks(frame.shape, landmarks_points).astype(np.float32)

        else:
            overlay_rgb = warped_overlay
            mask = np.ones_like(overlay_rgb, dtype=np.float32)

        # Crop the mask to the face region
        mask = mask[face.top():face.bottom(), face.left():face.right()]

        # Ensure mask and overlay_rgb are resized/cropped to match exactly
        overlay_rgb_resized = cv2.resize(overlay_rgb, (mask.shape[1], mask.shape[0]))

        # Crop the region of interest (ROI) from the frame
        roi = frame[face.top():face.bottom(), face.left():face.right()]

        # Convert overlay, mask, and roi to proper dtype for blending
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = mask.astype(np.float32)
        overlay_rgb_resized = overlay_rgb_resized.astype(np.float32)
        roi = roi.astype(np.float32)

        # Simple overlay using the mask
        blended = roi * (1.0 - mask) + overlay_rgb_resized * mask

        # Convert blended result back to uint8
        frame[face.top():face.bottom(), face.left():face.right()] = blended.astype(np.uint8)

    # Display the resulting frame
    cv2.imshow('Deep Fake Filter', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()