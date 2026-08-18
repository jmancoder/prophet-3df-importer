bl_info = {
    "name": "ADDON_NAME",
    "author": "AUTHOR_NAME",
    "description": "",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "File > Import",
    "category": "Import-Export",
}


from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from bpy.types import Operator, Context

from . import scene_3df


class IMPORT_OT_SCENE_3df(Operator, ImportHelper):
    """Load a 3DF scene."""
    bl_idname = "import_scene.prophet_3df"
    bl_label = "Import 3DF"
    filename_ext = ".3df"

    filter_glob: StringProperty(
        default="*.3df",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context: Context):
        in_path = Path(self.filepath)
        with open(in_path, "rb") as f:
            scene_data = scene_3df.read_3df(f)
        scene_3df.import_3df(scene_data, context)

        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_SCENE_3df.bl_idname,
                         text="Prophet 3DF (.3df)")


def register():
    bpy.utils.register_class(IMPORT_OT_SCENE_3df)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(IMPORT_OT_SCENE_3df)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
