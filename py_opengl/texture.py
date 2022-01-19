"""Texture
"""
from dataclasses import dataclass
from pathlib import Path

from OpenGL import GL
from PIL import Image

# pillow api ref
# https://pillow.readthedocs.io/en/stable/reference/index.html

class TextureError(Exception):
    '''Custom error for Texture'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Texture:
    width: float = 0.0
    height: float = 0.0
    texture_id: int = 0

    def compile(self, file_name: str) -> None:
        """Create and compile a texture for opengl to use within a shader

        Parameters
        ---
        file_name : str
            file name of image that should be located within
            the *./images* directory

        Raises
        ---
        TextureError
            texture file not found
        """
        file = Path(f'py_opengl/images/{file_name}').absolute()

        if not file.exists():
            raise TextureError('that texture was not found within images folder')

        # use pillow to open tetxure image file
        with Image.open(file.as_posix()) as im:
            self.texture_id = GL.glGenTextures(1)
            self.width, self.height = im.size
            border: int = 0
            level: int = 0

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                level,
                GL.GL_RGB,
                self.width,
                self.height,
                border,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                im.tobytes()
            )

    def clean(self) -> None:
        """Clean up texture from opengl
        """     
        GL.glDeleteTextures(1, self.texture_id)

    def use(self) -> None:
        """Use texture within opengl based on currently stored texture id
        """
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)