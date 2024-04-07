import cv2
import os


image_folder = 'Animation/Frames'
video_name = 'Animation/video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".bmp")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 1, 24, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()