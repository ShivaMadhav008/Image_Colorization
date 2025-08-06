!pip install opencv-python
 !pip install matplotlib
 !pip install scikit-image
 import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 from skimage import color
 from google.colab.patches import cv2_imshow
 from google.colab import files
 uploaded = files.upload()
 !ls
 net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'caffe_model.caffemodel')
 pts = np.load('pts_in_hull.npy')
 class8 = net.getLayerId('class8_ab')
 conv8 = net.getLayerId('conv8_313_rh')
 pts = pts.transpose().reshape(2, 313, 1, 1)
 net.getLayer(class8).blobs = [pts.astype(np.float32)]
 net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
 # Load and preprocess the grayscale image
 img_path = list(uploaded.keys())[0]
 frame = cv2.imread(img_path)
 scaled = frame.astype("float32") / 255.0
 lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
 resized = cv2.resize(lab, (224, 224))
 L = resized[:, :, 0]
 L -= 50  # mean-centering
 net.setInput(cv2.dnn.blobFromImage(L))
 ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
 ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
 L = lab[:, :, 0]
 colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
 colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
 colorized = np.clip(colorized, 0, 1)
 # Show original and colorized images
 plt.figure(figsize=(12, 6))
 plt.subplot(1, 2, 1)
 plt.title('Grayscale Input')
 plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 plt.subplot(1, 2, 2)
 plt.title('Colorized Output')
 plt.imshow((colorized * 255).astype(np.uint8))
 plt.show()
 colorized_output = (colorized * 255).astype(np.uint8)
 cv2.imwrite('colorized_output.png', cv2.cvtColor(colorized_output, cv2.COLOR_RGB2BGR))
 files.download('colorized_output.png')
