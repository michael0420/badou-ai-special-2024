import zoom
import cv2

img_zoom_out = zoom.zoom_image("lenna.png", 200, 200)
cv2.imshow('Zoom out', img_zoom_out)
img_zoom_in = zoom.zoom_image("lenna.png", 800, 800)
cv2.imshow('Zoom in', img_zoom_in)
cv2.waitKey(0)
