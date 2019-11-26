import requests

KERAS_REST_API_URL = "http://localhost:5000/classify"
IMAGE_PATH = "jemma.png"

image = open(IMAGE_PATH, "rb").read()
print(image)
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()

if r["success"]:
    # over the predictions
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
                                      result["probability"]))
else:
    print("Request failed")
