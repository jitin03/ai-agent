from fastapi import FastAPI

from routers import llm
from routers import run_prompt
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthy")
def health_check():
    return {'status': 'Healthy'}



app.include_router(llm.router)
app.include_router(run_prompt.router)

