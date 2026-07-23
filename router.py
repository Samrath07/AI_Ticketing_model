from fastapi import APIRouter, UploadFile, File, status, Depends, HTTPException
from schemas import FileMetaDataResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from database import get_db
import pandas as pd
import io,os

router = APIRouter(prefix="/upload", tags=["File Upload"])

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")

engine = create_engine(DATABASE_URL)

@router.post("/", response_model=FileMetaDataResponse, status_code=status.HTTP_201_CREATED)
async def upload_csv_file(file : UploadFile = File(...), db: Session = Depends(get_db)):

    if not file.filename.endswith('csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="contents not found csv file is empty")
        
        table_name = os.path.splitext(file.filename)[0].lower().replace(" ", "_").replace("-", "_")
        df.to_sql(name = table_name,con=engine, if_exists="replace", index=False)

        return {
            "message": f"Successfully created table '{table_name}' and inserted data.",
            "rows_inserted": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error occured file uploading the file: {str(e)}")


# @router.get('/train-pipeline/{table_name}')
# async def train_pipeline(table_name : str):
#     try:
#         df = pd.read_sql(f"Select * from {table_name}", con=engine)
#         required_cols = {'ticket description','ticket type', 'product purchased', 'ticket subject', 'ticket priority'}
        

