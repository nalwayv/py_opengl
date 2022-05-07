"""Texture
"""
from dataclasses import dataclass
from pathlib import Path

from OpenGL import GL
from PIL import Image


# ---


# pillow api ref
# https://pillow.readthedocs.io/en/stable/reference/index.html

class TextureError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq= False, repr= False, slots= True)
class Texture:
    texture_name: str
    _id: int= -1

    def __post_init__(self) -> None:
        """
        Raises
        ---
        TextureError
            texture file is not located
        """
        file: Path= Path(f'py_opengl/images/{self.texture_name}').absolute()

        if not file.exists():
            raise TextureError('that texture was not found within images folder')

        # use pillow to open tetxure image file
        with Image.open(file.as_posix()) as im:
            self._id= GL.glGenTextures(1)
            width, height= im.size
            border: int= 0
            level: int= 0

            GL.glBindTexture(GL.GL_TEXTURE_2D, self._id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                level,
                GL.GL_RGB,
                width,
                height,
                border,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                im.tobytes()
            )

    def clean(self) -> None:
        """Clean up texture from opengl
        """     
        GL.glDeleteTextures(1, self._id)

    def use(self) -> None:
        """Use texture within opengl based on currently stored texture id
        """
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._id)