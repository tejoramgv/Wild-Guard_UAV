"""Windows-only: build the optional thermal post-process material in UE 4.27.

Status: written against the UE 4.27 Python API but NOT executable in this
Linux environment. The validated thermal path is AirSim ImageType 7
(Infrared, object-ID greyscale) + configure_segmentation_ids.py. Use this
material only if you want a colour-ramped thermal look from a separate
SceneCapture2D.

Run inside the Unreal Editor Python console:
    py "ForestAirSim/Scripts/ue_thermal_setup.py"
"""
import unreal

MATERIAL_DIR = "/Game/Forest/Materials"
MATERIAL_NAME = "M_ThermalPostProcess"
USF_PATH = "ForestAirSim/Shaders/ThermalPostProcess.usf"


def _code():
    with open(USF_PATH) as handle:
        shader = handle.read()
    return shader + "\nreturn ThermalRamp(Stencil);\n"


def main():
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    library = unreal.MaterialEditingLibrary
    path = f"{MATERIAL_DIR}/{MATERIAL_NAME}"
    material = unreal.EditorAssetLibrary.load_asset(path)
    if material is None:
        material = tools.create_asset(MATERIAL_NAME, MATERIAL_DIR,
                                      unreal.Material, unreal.MaterialFactoryNew())

    library.delete_all_material_expressions(material)
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)

    stencil = library.create_material_expression(
        material, unreal.MaterialExpressionSceneTexture, -700, 0)
    try:
        stencil.set_editor_property(
            "scene_texture_id", unreal.SceneTextureId.PPI_CUSTOM_STENCIL)
    except Exception as exc:
        unreal.log_error(f"SceneTexture enum mismatch, adjust manually: {exc}")

    custom = library.create_material_expression(
        material, unreal.MaterialExpressionCustom, -350, 0)
    custom.set_editor_property("code", _code())
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    custom_input = unreal.CustomInput()
    custom_input.set_editor_property("input_name", "Stencil")
    custom.set_editor_property("inputs", [custom_input])
    library.connect_material_expressions(stencil, "Color", custom, "Stencil")
    library.connect_material_property(
        custom, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    library.recompile_material(material)
    unreal.EditorAssetLibrary.save_asset(path)
    unreal.log(f"created {path}")
    unreal.log("Create a Material Instance and add it to the post-process "
               "blendables of your thermal SceneCapture2D (or the AirSim "
               "InfraredCaptureComponent in BP_PIPCamera).")


if __name__ == "__main__":
    main()
