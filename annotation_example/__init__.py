# In the "my_package.__init__" submodule:
from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(
    conf=BeartypeConf(
        claw_skip_package_names=(
            "annotation_example.gradio_ui.mv_sam_callbacks",
            "annotation_example.gradio_ui.sv_sam_callbacks",
            "annotation_example.gradio_ui.person_annot_callbacks",
        )
    )
)
