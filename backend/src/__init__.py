from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.db.main import initdb
from src.auth.routers import auth_router
from fastapi.middleware.cors import CORSMiddleware
from src.mlapp.routers import ml_router
from fastapi.staticfiles import StaticFiles

version = 'v1'

origins = [
    "*"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("server is starting")
    await initdb()
    print("connected to database")
    yield
    print("server is stopping")

app = FastAPI(
    title='SensorGrid',
    description='A restful API for IoT devices to store data',
    version=version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount('/api/static/videos',StaticFiles(directory="videos"), name="videos")
app.mount('/api/static/images',StaticFiles(directory="images"), name="images")

app.include_router(router=auth_router,prefix='/api/{version}/auth', tags=["auth"])
app.include_router(router=ml_router,prefix='/api/{version}/mlapp', tags=["mlapp"])

