from PIL import Image

imgA = Image.open("imageA.png")
imgB = Image.open("imageB.png")
imgC = Image.open("imageC.png")

# Resize B and C to same height
target_height = 800
imgB = imgB.resize((int(imgB.width * target_height / imgB.height), target_height))
imgC = imgC.resize((int(imgC.width * target_height / imgC.height), target_height))

bottom_width = imgB.width + imgC.width

canvas_width = max(imgA.width, bottom_width)
canvas_height = imgA.height + target_height

canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

# Center top image
canvas.paste(imgA, ((canvas_width - imgA.width)//2, 0))
canvas.paste(imgB, (0, imgA.height))
canvas.paste(imgC, (imgB.width, imgA.height))

canvas.save("all_graphs_combined.pdf", "PDF", resolution=300.0)