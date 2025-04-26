"""
Label images
"""
# Imports the Google Cloud client library
from google.cloud import vision


def label_img(uri: str) -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = uri

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    return labels


if __name__ == "__main__":
    test_uri = "https://researchleap.com/wp-content/uploads/2021/01/people-different-ages-demos-ss-different-age-group.jpg"
    # test_uri = "https://i.redd.it/mv72p2fef7xe1.jpeg"
    lbls = [l.description for l in label_img(test_uri)]
    print(lbls)