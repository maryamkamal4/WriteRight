import base64
import cv2

def encode_images(canvas):
    try:
        # cv2.imshow('image2', image2)
        # # cv2.imshow('canvas', canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Convert images to base64
        # _, image2_encoded = cv2.imencode('.png', image2)
        _, canvas_encoded = cv2.imencode('.png', canvas)
        
        # Check if encoding was successful
        if not _:
            raise Exception("Image encoding failed")

        # Convert encoded image data to base64
        # image2_base64 = base64.b64encode(image2_encoded).decode('utf-8')
        canvas_base64 = base64.b64encode(canvas_encoded).decode('utf-8')

        return canvas_base64
    
    except Exception as e:
        print("Error during image encoding:", str(e))
        return "", ""
