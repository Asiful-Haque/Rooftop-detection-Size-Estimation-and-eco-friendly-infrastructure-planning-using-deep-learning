import os
import cv2
import numpy as np
from ultralytics import YOLO

# Define input and output folders
input_folder = '/home/dslab/Documents/Rooftop detection/test/images'
output_folder = 'segmented_images'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = YOLO('/home/dslab/Documents/Rooftop detection/runs/segment/train/weights/best.pt')

# Loop through images in the input folder
for image_name in os.listdir(input_folder):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        image_path = os.path.join(input_folder, image_name)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(img, conf=0.5)

        for i in results:
            try:
                for j in i.masks:  # Iterate through all masks in this Results object
                    # Extract all polygon vertices (xy holds all points)
                    points = np.array(j.xy, dtype=np.int32)

                    # Draw closed polygon using all points
                    cv2.polylines(img, [points], True, (0, 255, 255), 4)
            except:
                pass

        cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Save segmented image to output folder
        segmented_img_path = os.path.join(output_folder, f"segmented_{image_name}")
        # cv2.imwrite(segmented_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Segmented image saved: {segmented_img_path}")

cv2.destroyAllWindows()
print("All images segmented and saved successfully.")
