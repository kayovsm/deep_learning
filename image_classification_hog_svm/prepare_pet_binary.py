import os
import shutil

ROOT = os.path.dirname(__file__)
images_dir = os.path.join(ROOT, 'images')
ann_file = os.path.join(ROOT, 'annotations', 'list.txt')
pos_dir = os.path.join(ROOT, 'positive')
neg_dir = os.path.join(ROOT, 'negative')

os.makedirs(pos_dir, exist_ok=True)
os.makedirs(neg_dir, exist_ok=True)

cats = []
dogs = []

# lê o arquivo de anotações e separa nomes por espécie (1=gatos, 2=cães)
with open(ann_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        species = parts[1]
        if species == '1':
            cats.append(name + '.jpg')
        elif species == '2':
            dogs.append(name + '.jpg')

print(f'Found {len(cats)} cats and {len(dogs)} dogs in annotations')

# seleciona até 100 imagens de cada classe (ajuste `n` para alterar a quantidade)
n = 100
cats_sel = cats[:n]
dogs_sel = dogs[:n]

copied_c = 0
copied_d = 0

# copia imagens selecionadas para as pastas de destino
for fname in cats_sel:
    src = os.path.join(images_dir, fname)
    dst = os.path.join(pos_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied_c += 1

for fname in dogs_sel:
    src = os.path.join(images_dir, fname)
    dst = os.path.join(neg_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied_d += 1

print(f'Copied {copied_c} cat images to {pos_dir}')
print(f'Copied {copied_d} dog images to {neg_dir}')
