

def get_new_image_size(width, height, image_min_sid=600):
    if width<=height:
        f = float(image_min_sid) / width
        image_height = int(f / height)
        image_width = int(image_min_sid)
    else:
        f = float(image_min_sid) / height
        image_width = int(f / width)
        image_height = int(image_min_sid)
    return image_width, image_height