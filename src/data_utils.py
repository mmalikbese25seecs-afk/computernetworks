"""
data_utils.py
Helpers to (1) look at the SeaDronesSee folder and (2) convert it to YOLO format.
YOLO format = one .txt label file per image, one line per object:
    class_id  x_center  y_center  width  height     (all numbers between 0 and 1)
"""
import os, json, random
import cv2

IMG_EXT = ('.jpg', '.jpeg', '.png')


def show_tree(root, max_depth=3, max_files=3):
    """Print the folder structure so we can SEE how the dataset is organised."""
    root = root.rstrip('/')
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        print('  ' * depth + os.path.basename(dirpath) + '/   (' + str(len(filenames)) + ' files)')
        for f in sorted(filenames)[:max_files]:
            print('  ' * (depth + 1) + f)


def find_images(root):
    """Return every image path under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def find_coco_jsons(root):
    """Find annotation files in COCO format (a json with images/annotations/categories)."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.lower().endswith('.json'):
                continue
            p = os.path.join(dirpath, f)
            try:
                with open(p) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if isinstance(d, dict) and 'images' in d and 'annotations' in d and 'categories' in d:
                found.append(p)
    return sorted(found)


def _records_from_coco(json_path, by_name):
    """Turn one COCO json into a list of (image_path, [(class, cx, cy, w, h), ...])."""
    with open(json_path) as fh:
        d = json.load(fh)
    cats = sorted(d['categories'], key=lambda c: c['id'])
    keep = [c for c in cats if 'ignore' not in c['name'].lower()]     # drop the 'ignored' class
    cid_to_idx = {c['id']: i for i, c in enumerate(keep)}
    names = [c['name'] for c in keep]

    anns = {}
    for a in d['annotations']:
        if a['category_id'] in cid_to_idx:
            anns.setdefault(a['image_id'], []).append(a)

    records = []
    for im in d['images']:
        fname = im['file_name'].replace('\\', '/')
        candidates = by_name.get(os.path.basename(fname), [])
        if not candidates:
            continue
        path = candidates[0]
        for c in candidates:                       # prefer the path that ends with the json's file_name
            if c.replace('\\', '/').endswith(fname):
                path = c
                break
        W, H = im.get('width'), im.get('height')
        if not W or not H:
            H, W = cv2.imread(path).shape[:2]
        boxes = []
        for a in anns.get(im['id'], []):
            x, y, w, h = a['bbox']
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, W), min(y + h, H)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append((cid_to_idx[a['category_id']],
                          (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        if boxes:                                   # keep only images that have labels
            records.append((path, boxes))
    return records, names


def _records_from_yolo_txt(root, all_imgs):
    """Fallback: the dataset already has YOLO .txt labels next to / beside the images."""
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in all_imgs}
    records, max_id = [], 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext != '.txt' or stem not in by_stem:
                continue
            boxes = []
            for line in open(os.path.join(dirpath, f)):
                p = line.split()
                if len(p) == 5:
                    boxes.append((int(float(p[0])),) + tuple(float(v) for v in p[1:]))
                    max_id = max(max_id, int(float(p[0])))
            if boxes:
                records.append((by_stem[stem], boxes))
    return records, [f'class_{i}' for i in range(max_id + 1)]


def _save_resized(src, dst, max_side):
    """Copy an image, shrinking it so its longest side is at most max_side pixels (faster training)."""
    img = cv2.imread(src)
    if img is None:
        return False
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def build_yolo_dataset(root, out_dir, max_train=2000, max_val=400, max_side=1280, seed=0, reuse=True):
    """
    Main function: read the raw dataset, pick a subset, write it in YOLO format to out_dir.
    Both teammates call this with the SAME arguments, so both get the SAME train/val split.
    """
    yaml_path = os.path.join(out_dir, 'data.yaml')
    if reuse and os.path.exists(yaml_path):
        print('Dataset already built, reusing', out_dir)
        return load_info(out_dir)

    rnd = random.Random(seed)
    all_imgs = find_images(root)
    print('images found:', len(all_imgs))
    by_name = {}
    for p in all_imgs:
        by_name.setdefault(os.path.basename(p), []).append(p)

    jsons = find_coco_jsons(root)
    print('COCO annotation files found:', jsons)
    splits = {'train': [], 'val': []}
    pool = []
    if jsons:
        names = None
        for jp in jsons:
            recs, nm = _records_from_coco(jp, by_name)
            names = names or nm
            low = os.path.basename(jp).lower() + '/' + os.path.dirname(jp).lower()
            print(f'  {jp}: {len(recs)} labelled images')
            if 'test' in low:
                continue
            if 'val' in low:
                splits['val'] += recs
            elif 'train' in low:
                splits['train'] += recs
            else:
                pool += recs
    else:
        print('No COCO json found -> trying YOLO .txt labels')
        recs, names = _records_from_yolo_txt(root, all_imgs)
        for r in recs:
            low = r[0].lower()
            if '/val' in low:
                splits['val'].append(r)
            elif '/train' in low:
                splits['train'].append(r)
            else:
                pool.append(r)

    pool.sort(key=lambda r: r[0])
    rnd.shuffle(pool)
    if not splits['val']:                            # no official val set -> take 15% of the data
        cut = int(len(pool) * 0.15) if pool else int(len(splits['train']) * 0.15)
        if pool:
            splits['val'], pool = pool[:cut], pool[cut:]
        else:
            splits['train'].sort(key=lambda r: r[0]); rnd.shuffle(splits['train'])
            splits['val'], splits['train'] = splits['train'][:cut], splits['train'][cut:]
    splits['train'] += pool
    if not splits['train'] or not splits['val']:
        raise RuntimeError('Could not find labelled images. Run show_tree(DATA_ROOT) and send the output to Claude.')

    for split, limit in [('train', max_train), ('val', max_val)]:
        recs = sorted(splits[split], key=lambda r: r[0])
        if len(recs) > limit:
            recs = sorted(rnd.sample(recs, limit), key=lambda r: r[0])   # random subset, kept in file order
        os.makedirs(f'{out_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{out_dir}/labels/{split}', exist_ok=True)
        n = 0
        for path, boxes in recs:
            stem = os.path.splitext(os.path.basename(path))[0]
            if not _save_resized(path, f'{out_dir}/images/{split}/{stem}.jpg', max_side):
                continue
            with open(f'{out_dir}/labels/{split}/{stem}.txt', 'w') as f:
                f.write('\n'.join(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}' for c, cx, cy, w, h in boxes))
            n += 1
        print(f'{split}: wrote {n} images')

    with open(yaml_path, 'w') as f:
        f.write(f'path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n')
        for i, nm in enumerate(names):
            f.write(f'  {i}: {nm}\n')
    return load_info(out_dir)


def load_info(out_dir):
    import yaml
    yaml_path = os.path.join(out_dir, 'data.yaml')
    d = yaml.safe_load(open(yaml_path))
    names = d['names'] if isinstance(d['names'], dict) else dict(enumerate(d['names']))
    n_tr = len(os.listdir(f'{out_dir}/images/train'))
    n_va = len(os.listdir(f'{out_dir}/images/val'))
    return {'yaml': yaml_path, 'names': names, 'n_train': n_tr, 'n_val': n_va, 'dir': out_dir}
