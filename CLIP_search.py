import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PySimpleGUI as sg
from pathlib import Path


layout = [
    [sg.Input(visible=True, enable_events=False, key="-FOLDER-"), sg.FolderBrowse()],
    [sg.Text("Query"), sg.InputText(key="-QUERY-"), sg.Submit()],
    [sg.Text("Batch Size"), sg.Input(key="-BATCH_SIZE-")]
]

window = sg.Window("Choose folders to search", layout)
event, values = window.read()
window.close()

batch_size = int(values["-BATCH_SIZE-"]) if values["-BATCH_SIZE-"] else 10
vision_encoder = tf.keras.models.load_model('vision_encoder')
text_encoder = tf.keras.models.load_model('hm/text_encoder')

image_paths = []
for path in Path(values["-FOLDER-"]).rglob('*'):
    if path.suffix in [".jpg"]:
        image_paths.append(str(path.resolve()))

print(image_paths)
print(f"Number of images: {len(image_paths)}")


def read_image(image_path):
    image_array = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return tf.image.resize(image_array, (299, 299))


image_embeddings = vision_encoder.predict(
    tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(batch_size),
    verbose=1,
)


def find_matches(image_embeddings, queries, k=9, normalize=True):
    query_embedding = text_encoder(tf.convert_to_tensor(queries))
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    return [[image_paths[idx] for idx in indices] for indices in results]


query = values["-QUERY-"]
matches = find_matches(image_embeddings, [query], normalize=True)[0]

plt.figure(figsize=(20, 20))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(mpimg.imread(matches[i]))
    plt.axis("off")
plt.show()
