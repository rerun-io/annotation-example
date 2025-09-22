from pathlib import Path

import rerun.blueprint as rrb


def create_depth_views(parent_log_path: Path, camera_index: int) -> rrb.Tabs:
    """
    Create depth visualization tabs for a specific camera.

    Args:
        parent_log_path: Parent log path for the camera views
        camera_index: Index of the camera to create depth views for

    Returns:
        Tabs blueprint containing depth and filtered depth views
    """
    depth_views: rrb.Tabs = rrb.Tabs(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/depth",
                contents=[
                    "+ $origin/**",
                ],
                name="Depth",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/filtered_depth",
                contents=[
                    "+ $origin/**",
                ],
                name="Filtered Depth",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/refined_depth",
                contents=[
                    "+ $origin/**",
                ],
                name="MoGe Depth",
            ),
        ],
        active_tab=2,
    )
    return depth_views


def create_camera_row(parent_log_path: Path, camera_index: int) -> rrb.Horizontal:
    """
    Create a single camera row with 3 views: content, depth, and confidence.

    Args:
        parent_log_path: Parent log path for the camera views
        camera_index: Index of the camera to create views for

    Returns:
        Horizontal blueprint containing pinhole content, depth views, and confidence map
    """
    camera_row: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/image",
                contents=[
                    "+ $origin/**",
                ],
                name="Image Content",
            ),
            create_depth_views(parent_log_path, camera_index),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/confidence",
                contents=[
                    "+ $origin/**",
                ],
                name="Confidence Map",
            ),
        ]
    )
    return camera_row


def chunk_cameras(num_cameras: int, chunk_size: int = 4) -> list[range]:
    """
    Group cameras into chunks of specified size.

    Args:
        num_cameras: Total number of cameras
        chunk_size: Maximum cameras per chunk (default 4)

    Returns:
        List of ranges representing camera chunks
    """
    chunks: list[range] = [range(i, min(i + chunk_size, num_cameras)) for i in range(0, num_cameras, chunk_size)]
    return chunks


def create_tabbed_camera_view(parent_log_path: Path, num_cameras: int) -> rrb.Tabs:
    """
    Create tabbed interface grouping cameras by 4s.

    Args:
        parent_log_path: Parent log path for the camera views
        num_cameras: Total number of cameras to display

    Returns:
        Tabs blueprint with each tab containing up to 4 camera rows
    """
    camera_chunks: list[range] = chunk_cameras(num_cameras)

    tabs: list[rrb.Vertical] = []
    for camera_range in camera_chunks:
        # Create camera rows for this chunk
        camera_rows: list[rrb.Horizontal] = [create_camera_row(parent_log_path, i) for i in camera_range]

        # Create tab name
        if camera_range.start + 1 == camera_range.stop:
            tab_name: str = f"Camera {camera_range.start + 1}"
        else:
            tab_name = f"Cameras {camera_range.start + 1}-{camera_range.stop}"

        # Create tab content
        tab_content: rrb.Vertical = rrb.Vertical(contents=camera_rows, name=tab_name)
        tabs.append(tab_content)

    tabbed_view: rrb.Tabs = rrb.Tabs(contents=tabs, name="Depths Tab")
    return tabbed_view


def create_view_container(parent_log_path: Path, num_images: int, show_videos: bool = False) -> rrb.Container:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/filtered_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/refined_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/confidence" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/image" for i in range(num_images)],
        ],
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
    )

    # Create tabbed view that supports any number of cameras
    view_2d: rrb.Tabs = create_tabbed_camera_view(parent_log_path, num_images)
    if show_videos:
        view_2d_videos: rrb.Grid = rrb.Grid(
            contents=[
                rrb.Spatial2DView(origin=f"{parent_log_path}/exo/camera_{i}/pinhole/video", name=f"Video {i + 1}")
                for i in range(num_images)
            ],
            name="Videos Tab",
        )
        view_2d = rrb.Tabs(view_2d, view_2d_videos, active_tab=1)

    final_view: rrb.Horizontal = rrb.Horizontal(contents=[view3d, view_2d], column_shares=[3, 2])
    return final_view
