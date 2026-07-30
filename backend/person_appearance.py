"""Face-scoped image crops used for person appearance evidence."""


def expanded_person_crop(image, bbox, head_scale=1.35, body_height_scale=4.5):
    """Return a bounded head-and-upper-body crop and its source coordinates.

    The supplied box is a detected face. The crop keeps the face centered and
    extends below it to capture clothing, without silently falling back to the
    complete scene when the source image is too small.
    """
    if len(bbox or []) != 4:
        raise ValueError("face bounding box must contain four coordinates")
    left, top, right, bottom = (float(value) for value in bbox)
    face_width, face_height = right - left, bottom - top
    if face_width < 2 or face_height < 2:
        raise ValueError("face bounding box is too small")
    center_x = (left + right) / 2
    crop_width = face_width * head_scale
    crop_left = max(0, int(round(center_x - crop_width / 2)))
    crop_right = min(image.width, int(round(center_x + crop_width / 2)))
    crop_top = max(0, int(round(top - face_height * 0.35)))
    crop_bottom = min(image.height, int(round(bottom + face_height * body_height_scale)))
    if crop_right - crop_left < 24 or crop_bottom - crop_top < 48:
        raise ValueError("person appearance crop is too small")
    return image.crop((crop_left, crop_top, crop_right, crop_bottom)), [crop_left, crop_top, crop_right, crop_bottom]
