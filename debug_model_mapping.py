import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

IDX_PATH = Path('backend/models/orthovision_model.class_indices.json')

IMG_SIZE = (384, 384)


def prep(p: Path):
    img = tf.keras.preprocessing.image.load_img(str(p), target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def main():
    class_indices = json.loads(IDX_PATH.read_text(encoding='utf-8'))
    idx_to_class = {int(v): str(k) for k, v in class_indices.items()}

    print('class_indices:', class_indices)

    model_paths = [
        Path('backend/models/orthovision_model.keras'),
        Path('backend/models/best_model.keras'),
    ]

    frac_paths = list(Path('dataset2/train/fracture').glob('*.jpg')) + list(
        Path('dataset2/train/fracture').glob('*.png')
    )
    norm_paths = list(Path('dataset2/train/normal').glob('*.jpg')) + list(
        Path('dataset2/train/normal').glob('*.png')
    )

    for mp in model_paths:
        if not mp.exists():
            continue
        print('\n' + '=' * 60)
        print('MODEL:', mp.as_posix())
        print('=' * 60)
        model = tf.keras.models.load_model(mp)

        for label, paths in [('fracture', frac_paths), ('normal', norm_paths)]:
            if not paths:
                print('No images found for', label)
                continue
            p = random.choice(paths)
            probs = model.predict(prep(p), verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            print('\ntrue:', label, 'file:', p.name)
            print('probs:', [float(x) for x in probs])
            print('pred_idx:', pred_idx, 'pred_class:', idx_to_class.get(pred_idx))


if __name__ == '__main__':
    main()
