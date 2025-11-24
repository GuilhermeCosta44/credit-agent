from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles # Importante
from fastapi.responses import FileResponse # Importante
from app.models import ModelingRequest
from app.services.generator import generate_notebook_json
import datetime
import os

app = FastAPI(title="Credit Modeling Agent")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("app/static/index.html")

@app.post("/generate-notebook")
async def generate_notebook_endpoint(request: ModelingRequest):
    try:
        notebook_content = generate_notebook_json(request)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"AutoModel_{request.target_column}_{timestamp}.ipynb"
        
        return Response(
            content=notebook_content,
            media_type="application/x-ipynb+json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        print(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)