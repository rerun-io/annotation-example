import uuid

import rerun as rr


def get_recording(recording_id: uuid.UUID, application_id: str = "Manual Hand Annotation") -> rr.RecordingStream:
    return rr.RecordingStream(application_id=application_id, recording_id=recording_id)
