from pydantic import BaseModel

class ModelResponse(BaseModel):
    priority: str
    classification: int
    department : str
    confidence_score: int


class FileMetaDataResponse(BaseModel):
    unique_name : str
    original_name: str
    content_type : str
    file_path : str

    class Config:
        from_attributes = True



    