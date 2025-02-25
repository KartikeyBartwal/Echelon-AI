from fastapi import FastAPI
from utils.utils import generate_scenario

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

@app.get("/get_scenario")
async def get_scenario():
    scenario = await generate_scenario()
    return {"scenario": scenario}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)