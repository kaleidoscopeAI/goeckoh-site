class GifImageFormatter(ImageFormatter):
    """
    Create a GIF image from source code. This uses the Python Imaging Library to
    generate a pixmap from the source code.

    .. versionadded:: 1.0
    """

    name = 'img_gif'
    aliases = ['gif']
    filenames = ['*.gif']
    default_image_format = 'gif'


class JpgImageFormatter(ImageFormatter):
    """
    Create a JPEG image from source code. This uses the Python Imaging Library to
    generate a pixmap from the source code.

    .. versionadded:: 1.0
    """

    name = 'img_jpg'
    aliases = ['jpg', 'jpeg']
    filenames = ['*.jpg']
    default_image_format = 'jpeg'


class BmpImageFormatter(ImageFormatter):
    """
    Create a bitmap image from source code. This uses the Python Imaging Library to
    generate a pixmap from the source code.

    .. versionadded:: 1.0
    """

    name = 'img_bmp'
    aliases = ['bmp', 'bitmap']
    filenames = ['*.bmp']
    default_image_format = 'bmp'


