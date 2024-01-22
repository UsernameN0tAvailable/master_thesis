from PIL import Image

def build_image_from_tiles(name: str):
    image_size = (912, 912)  # Replace with the actual size of each patch
    grid_size = (4, 4)
    new_image = Image.new('RGB', (image_size[0] * grid_size[0], image_size[1] * grid_size[1]))

    for h in range(6):
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                # Load image (make sure to replace with your actual file names)
                image_path = f'/home/me/Uni/attention_maps/att_{y}_{x}/heads/{h}.png'
                patch = Image.open(image_path)

                # Paste the image at the correct position
                new_image.paste(patch, (y * image_size[0], x * image_size[1]))

        # Save the new image
        new_image.save(f"head_{h}_map.png")



if __name__ == "__main__":

    build_image_from_tiles("")
