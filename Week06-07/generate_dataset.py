import random
from PIL import Image
from tqdm import tqdm


def generate_image(id, neither, sheep, coke, base, mode):
    try:
        neither_image = Image.open(neither)
    except:
        return
    neither_image = neither_image.resize((1000, 1000))
    neither_w, neither_h = neither_image.size

    labels = []

    if sheep is not None:
        sheep_image = Image.open(sheep)
        sheep_w, sheep_h = sheep_image.size
        scale = random.random() + 0.5
        paste_w, paste_h = int(sheep_w * scale), int(sheep_h * scale)
        top_x, top_y = random.randint(0, neither_w - paste_w), random.randint(0, neither_h - paste_h)
        sheep_image = sheep_image.resize((paste_w, paste_h))
        neither_image.paste(
            sheep_image, (top_x, top_y), sheep_image
        )
        labels.append(f"0 {(top_x + paste_w // 2) / neither_w} {(top_y + paste_h // 2) / neither_h} {paste_w / neither_w} {paste_h / neither_h}\n")

    if coke is not None:
        coke_image = Image.open(coke)
        coke_w, coke_h = coke_image.size
        scale = random.random() + 0.5
        paste_w, paste_h = int(coke_w * scale), int(coke_h * scale)
        top_x, top_y = random.randint(0, neither_w - paste_w), random.randint(0, neither_h - paste_h)
        coke_image = coke_image.resize((paste_w, paste_h))
        neither_image.paste(
            coke_image, (top_x, top_y), coke_image
        )
        labels.append(f"1 {(top_x + paste_w // 2) / neither_w} {(top_y + paste_h // 2) / neither_h} {paste_w / neither_w} {paste_h / neither_h}\n")

    if len(labels) > 0:
        file = open(f'{base}/labels/{mode}/{id}.txt', 'w')
        file.writelines(labels)
    
    neither_image.resize((500, 500))
    neither_image.save(f'{base}/images/{mode}/{id}.jpg')


num_examples = 10_000

for id in tqdm(range(num_examples)):
    background = random.randint(201, 299)
    background = f'model_images/neither/{background}.png'
    
    sheep = random.randint(1, 14)
    if sheep > 12 or sheep == 5:
        sheep = None
    else:
        sheep = f'model_images/sheep/sheep_{sheep}-removebg-preview.png'
    coke = random.randint(1, 14)
    if coke > 12:
        coke = None
    else:
        coke = f'model_images/coke/coke_{coke}.png'

    generate_image(
        id, background, sheep, coke, './generated_dataset', 'train'
    )

num_examples = 1_000

for id in tqdm(range(num_examples)):
    background = random.randint(201, 299)
    background = f'model_images/neither/{background}.png'
    
    sheep = random.randint(1, 18)
    if sheep > 12 or sheep == 5:
        sheep = None
    else:
        sheep = f'model_images/sheep/sheep_{sheep}-removebg-preview.png'
    coke = random.randint(1, 18)
    if coke > 12:
        coke = None
    else:
        coke = f'model_images/coke/coke_{coke}.png'

    generate_image(
        id, background, sheep, coke, './generated_dataset', 'val'
    )