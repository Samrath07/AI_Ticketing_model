from fastapi import FastAPI,File, UploadFile
from router import upload_csv_file

app = FastAPI(
    title="Smart Ticket API", 
    description="A production-ready organized API",
    version="1.0.0"
    )

@app.post('/')
async def csv_file_upload(file : UploadFile = File(...)):
    return await upload_csv_file(file)







