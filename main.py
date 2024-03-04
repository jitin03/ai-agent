from fastapi import FastAPI

from routers import llm
from routers import run_prompt

app = FastAPI()




@app.get("/healthy")
def health_check():
    return {'status': 'Healthy'}



app.include_router(llm.router)
app.include_router(run_prompt.router)

