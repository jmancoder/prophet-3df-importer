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
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator, Context

from .reader_3df import Reader3DF
from .importer_3df import Importer3DF


class IMPORT_OT_SCENE_3df(Operator, ImportHelper):
    """Load a 3DF file."""

    bl_idname = "import_scene.prophet_3df"
    bl_label = "Import 3DF"
    filename_ext = ".3df"

    filter_glob: StringProperty(
        default="*.3df",
        options={"HIDDEN"},
        maxlen=255,
    )

    platform: EnumProperty(
        name="Platform",
        description="The platform of the game that the 3DF file is from.",
        items=(
            ("PS2", "PS2", "Playstation 2"),
            ("PC", "PC", "Windows"),
            ("WII", "Wii", "Nintendo Wii"),
            ("DS", "DS", "Nintendo DS"),
        ),
        default="PC",
    )

    import_anims: BoolProperty(
        name="Import Animations",
        description="Import animation tracks from the file. WARNING: Currently broken.",
        default=False,
    )

    def execute(self, context: Context):
        in_path = Path(self.filepath)
        with open(in_path, "rb") as f:
            reader = Reader3DF(self.platform)
            scene_data = reader.read_scene_from_file(f)

        importer = Importer3DF(context, self.import_anims)
        importer.import_scene(scene_data)

        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_SCENE_3df.bl_idname, text="Prophet 3DF (.3df)")


def register():
    bpy.utils.register_class(IMPORT_OT_SCENE_3df)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(IMPORT_OT_SCENE_3df)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
