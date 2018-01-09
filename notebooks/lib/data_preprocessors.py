from PIL import Image

def scale(image, max_size):
    """
    image is a PIL Image object
    max_size is a tuple of ints

    resize 'image' to 'max_size' keeping the aspect ratio 
    and place it in center of white 'max_size' image 
    """
    im_aspect = float(image.size[0])/float(image.size[1])
    out_aspect = float(max_size[0])/float(max_size[1])
    if im_aspect >= out_aspect:
        scaled = image.resize((max_size[0], int((float(max_size[0])/im_aspect) + 0.5)))
    else:
        scaled = image.resize((int((float(max_size[1])*im_aspect) + 0.5), max_size[1]))
 
    offset = (((max_size[0] - scaled.size[0]) // 2), ((max_size[1] - scaled.size[1]) // 2))
    back = Image.new("1", max_size, "white")
    back.paste(scaled, offset)
    return back