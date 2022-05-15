"""Texture
"""
from pathlib import Path

from OpenGL import GL
from PIL import Image


# ---


# pillow api ref
# https://pillow.readthedocs.io/en/stable/reference/index.html

class TextureError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Texture:

    __slots__= ('texture', '_id')

    def __init__(self, texture_file_name: str) -> None:
        """
        Raises
        ---
        TextureError
            texture file is not located
        """

        self.texture: str= texture_file_name
        self._id= GL.glGenTextures(1)

        file: Path= Path(f'py_opengl/images/{self.texture}').absolute()

        if not file.exists():
            GL.glDeleteTextures(1, self._id)
            raise TextureError('that texture was not found within images folder')

        # use pillow to open tetxure image file
        with Image.open(file.as_posix()) as im:
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

    def bind(self) -> None:
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._id)

    def unbind(self) -> None:
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def delete(self) -> None:
        """Clean up texture from opengl
        """     
        GL.glDeleteTextures(1, self._id)