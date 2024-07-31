import cv2
import numpy as np
import os

# Load the image of the celebrity
overlay_image_path = 'cristiano_ronaldo.png'  # Make sure to have this image in the same directory

# Verify the file exists
if not os.path.exists(overlay_image_path):
    raise FileNotFoundError(f"The file {overlay_image_path} does not exist. Please check the file path.")

overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

# Validate that the image was loaded
if overlay_image is None:
    raise ValueError(f"Failed to load the image from {overlay_image_path}. Please check the file path and integrity.")

# Start the video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize overlay image to fit the frame
    frame_height, frame_width = frame.shape[:2]
    overlay_height, overlay_width = overlay_image.shape[:2]

    scale_ratio = min(frame_width / overlay_width, frame_height / overlay_height)
    new_overlay_size = (int(overlay_width * scale_ratio), int(overlay_height * scale_ratio))
    resized_overlay_image = cv2.resize(overlay_image, new_overlay_size, interpolation=cv2.INTER_AREA)

    overlay_y, overlay_x = frame_height // 2 - new_overlay_size[1] // 2, frame_width // 2 - new_overlay_size[0] // 2

    # Extract the alpha channel and create masks for overlay
    if resized_overlay_image.shape[2] == 4:
        alpha_channel = resized_overlay_image[:, :, 3]
        overlay_rgb = resized_overlay_image[:, :, :3]

        mask = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)
        mask = mask / 255.0
    else:
        overlay_rgb = resized_overlay_image
        mask = np.ones_like(overlay_rgb, dtype=np.float32)

    # Convert mask and overlay_rgb to proper dtype for uint8 operation
    mask = mask.astype(np.float32)
    overlay_rgb = overlay_rgb.astype(np.float32)
    roi = frame[overlay_y:overlay_y + new_overlay_size[1], overlay_x:overlay_x + new_overlay_size[0]].astype(np.float32)

    # Simple overlay using the mask
    blended = roi * (1.0 - mask) + overlay_rgb * mask

    # Convert blended result back to uint8
    frame[overlay_y:overlay_y + new_overlay_size[1], overlay_x:overlay_x + new_overlay_size[0]] = blended.astype(np.uint8)

    # Display the resulting frame
    cv2.imshow('Deep Fake Filter', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()