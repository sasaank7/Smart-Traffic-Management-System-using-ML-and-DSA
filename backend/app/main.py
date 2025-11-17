"""
Smart Traffic Management System - Main FastAPI Application
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.core.config import settings
from app.core.database import init_db, close_db
from app.api import routes, traffic, routing, signals, auth, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    print("🚀 Starting Smart Traffic Management System...")
    await init_db()
    print("✅ Database initialized")

    yield

    # Shutdown
    print("🛑 Shutting down...")
    await close_db()
    print("✅ Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="Smart Traffic Management API",
    description="Production-ready traffic management system with ML and DSA integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# Include routers
app.include_router(routes.router, tags=["General"])
app.include_router(traffic.router, prefix="/api/v1/traffic", tags=["Traffic"])
app.include_router(routing.router, prefix="/api/v1/routing", tags=["Routing"])
app.include_router(signals.router, prefix="/api/v1/signals", tags=["Traffic Signals"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Traffic Management System API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=settings.API_WORKERS if not settings.DEBUG else 1,
    )
