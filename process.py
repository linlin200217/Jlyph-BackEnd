import math

from PIL import Image


def process_to_circle(image: Image.Image, number: int):
    petal_image = image.convert("RGBA")

    # 获取花瓣的大小
    petal_width, petal_height = petal_image.size

    # 创建一个新的背景图像
    _image = Image.new('RGBA', (500, 500))

    # 计算旋转的中心点
    circle_center_x, circle_center_y = 250, 250
    circle_radius = 150

    # 计算花瓣的底部中心点
    petal_center_x, petal_center_y = petal_width / 2, petal_height / 2

    def place_petal_at_angle(_degrees):
        # 根据角度计算放置点
        radians = math.radians(_degrees)
        x_offset = circle_radius * math.cos(radians)
        y_offset = circle_radius * math.sin(radians)
        x = circle_center_x + x_offset - petal_center_x
        y = circle_center_y + y_offset - petal_center_y
        # 将花瓣粘贴到计算出的位置
        rotated_petal = petal_image.rotate(-_degrees, resample=Image.BICUBIC, center=(petal_center_x, petal_center_y))
        _image.paste(rotated_petal, (int(x), int(y)), rotated_petal)

    for i in range(number):
        degrees = i * (360 / number)
        place_petal_at_angle(degrees)
    return _image


def process_to_radiation(image: Image.Image, number: int):
    def get_petal_bottom_coordinates(_image):
        """获取花瓣最底部的坐标"""
        for y in range(_image.height - 1, 0, -1):
            for x in range(_image.width):
                *_, alpha = _image.getpixel((x, y))  # 获取alpha通道的值
                if alpha > 0:
                    return x, y
        return None, None

    def rotate_image_at_point(_image, _degrees, x, y, expand=True):
        """以(x, y)为中心点旋转图像，并确保所有内容都适应图像"""
        if expand:
            # 将花瓣置于一个更大的透明背景上
            tmp_img = Image.new('RGBA', (_image.width * 3, _image.height * 3), (255, 255, 255, 0))
            tmp_img.paste(_image, (_image.width, _image.height), _image)
            x += _image.width
            y += _image.height
            rotated_img = tmp_img.rotate(-_degrees, resample=Image.BICUBIC, center=(x, y))
        else:
            rotated_img = _image.rotate(-_degrees, resample=Image.BICUBIC, center=(x, y))
        return rotated_img

    def get_flower_bounds(_image):
        """获取花的边界"""
        left, top, right, bottom = _image.width, _image.height, 0, 0

        for y in range(_image.height):
            for x in range(_image.width):
                _, _, _, alpha = _image.getpixel((x, y))
                if alpha > 0:  # 如果当前像素不是完全透明的
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x)
                    bottom = max(bottom, y)

        return left, top, right, bottom

    def center_flower(_image):
        left, top, right, bottom = get_flower_bounds(_image)

        # 计算花的中心
        flower_center_x = (left + right) // 2
        flower_center_y = (top + bottom) // 2

        # 计算偏移量
        dx = _image.width // 2 - flower_center_x
        dy = _image.height // 2 - flower_center_y

        # 创建新的透明背景图像
        result_image = Image.new('RGBA', _image.size, (255, 255, 255, 0))

        # 在新的位置上粘贴花
        result_image.paste(_image, (dx, dy), _image)

        return result_image

    # 获取花瓣最底部的坐标
    petal_bottom_x, petal_bottom_y = get_petal_bottom_coordinates(image)

    # 创建一个新的背景图像
    background_size = max(image.width, image.height) * 3
    flower_image4 = Image.new('RGBA', (background_size, background_size), (255, 255, 255, 0))
    bg_center = (background_size // 2, background_size // 2)  # 计算背景的中心点

    # 将20个花瓣放置在旋转的位置
    for i in range(number):
        degrees = i * (360 / number)
        rotated_petal = rotate_image_at_point(image, degrees, petal_bottom_x, petal_bottom_y, expand=True)
        offset_x = bg_center[0] - rotated_petal.width // 2
        offset_y = bg_center[1] - rotated_petal.height // 2
        flower_image4.paste(rotated_petal, (offset_x, offset_y), rotated_petal)
    return center_flower(flower_image4)


def process_to_transverse(image: Image.Image, shift_amount: int, number: int):
    new_width = image.width + shift_amount * (number - 1)

    # 新建一个调整尺寸后的透明背景图像
    output_image = Image.new('RGBA', (new_width, image.height), (0, 0, 0, 0))

    for i in range(number):
        # 将原始图片粘贴到新图像的偏移位置
        output_image.paste(image, (i * shift_amount, 0), image)

    return output_image


def process_to_vertical(image: Image.Image, shift_amount: int, number: int):
    # 调整输出图片的高度
    new_height = image.height + shift_amount * (number - 1)

    # 新建一个调整尺寸后的透明背景图像
    output_image = Image.new('RGBA', (image.width, new_height), (0, 0, 0, 0))

    for i in range(number):
        # 将原始图片粘贴到新图像的偏移位置
        output_image.paste(image, (0, i * shift_amount), image)

    return output_image


def process_to_combination(main_image, sub_image, sub_of_num, circle_center, circle_radius) -> Image.Image:
    width, height = sub_image.size
    new_width = int(width * 0.2)
    new_height = int(height * 0.2)
    sub_image = sub_image.resize((new_width, new_height))
    new_image = main_image.copy()

    angle_step = 360 / sub_of_num
    for i in range(sub_of_num):
        angle = math.radians(i * angle_step)
        x = circle_center[0] + circle_radius * math.cos(angle) - sub_image.width // 2
        y = circle_center[1] + circle_radius * math.sin(angle) - sub_image.height // 2
        new_image.paste(sub_image, (int(x), int(y)), sub_image)
    return new_image


def scale_image(image: Image.Image):
    width, height = image.size

    # 定义花的位置和大小
    flower_x = width // 4
    flower_y = height // 4
    flower_width = width // 2
    flower_height = height // 2

    # 获取花区域
    flower_region = image.crop((flower_x, flower_y, flower_x + flower_width, flower_y + flower_height))

    # 放大花区域
    new_flower_width = int(flower_width * 2)
    new_flower_height = int(flower_height * 2)
    enlarged_flower = flower_region.resize((new_flower_width, new_flower_height), Image.ANTIALIAS)

    # 创建新的图片，透明背景
    new_image = Image.new('RGBA', (width, height))
    new_image.paste(image, (0, 0))
    new_image.paste(enlarged_flower, (
        flower_x - (new_flower_width - flower_width) // 2, flower_y - (new_flower_height - flower_height) // 2),
                    enlarged_flower)
    return new_image
