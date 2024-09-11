
def apply_image_enhancements(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_slider.value)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_slider.value)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(highlights_slider.value)
    pil_img = ImageEnhance.Color(pil_img).enhance(white_slider.value)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(black_slider.value)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def update_images(option=None):
    global image_path
    blur_radius = blur_radius_slider.value
    cutout_threshold = cutout_threshold_slider.value

    image = cv2.imread(image_path)
    black_background = np.zeros_like(image)

    if option == 'Background Blur':
        ori = Image.open(image_path)
        blurr = ori.filter(ImageFilter.GaussianBlur(blur_radius))
        blurr_np = cv2.cvtColor(np.array(blurr), cv2.COLOR_RGB2BGR)
        cv2.imwrite('blured_background/blurr1.png', blurr_np)
        mask = segmentor.removeBG(image, black_background, cutThreshold=cutout_threshold)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        final_segmented_image = cv2.bitwise_and(image, image, mask=cleaned_mask)
        cv2.imwrite('image_cutout/segmented_image.png', final_segmented_image)
        blurr_image = cv2.imread('blured_background/blurr1.png')
        segmented_image = cv2.imread('image_cutout/segmented_image.png')
        if blurr_image.shape != segmented_image.shape:
            blurr_image = cv2.resize(blurr_image, (segmented_image.shape[1], segmented_image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        mask_inv = cv2.bitwise_not(cv2.threshold(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1])
        merged_image = cv2.add(cv2.bitwise_and(blurr_image, blurr_image, mask=mask_inv), cv2.bitwise_and(segmented_image, segmented_image, mask=cv2.threshold(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]))
        merged_image = apply_image_enhancements(merged_image)
        cv2.imwrite('final_image/merged_image.png', merged_image)
        img_filename = 'final_image/merged_image.png'
