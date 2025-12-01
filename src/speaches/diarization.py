from pydantic import BaseModel, ConfigDict

from speaches.audio import Audio

class KnownSpeaker(BaseModel):
    name: str
    audio: Audio

    model_config = ConfigDict(arbitrary_types_allowed=True)
