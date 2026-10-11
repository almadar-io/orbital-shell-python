"""
Generated Route Handlers

FastAPI route handlers generated from .orb schema transitions.

This module provides a register_routes function that adds all generated
routes to the FastAPI application.
"""
import os
from fastapi import FastAPI


def register_routes(app: FastAPI) -> None:
    """
    Register all generated routes with the FastAPI application.

    This function is called from main.py to add all event handlers
    generated from the .orb schema.

    Args:
        app: FastAPI application instance
    """
    # Import example routes (can be removed in production)
    from .example import router as example_router

    app.include_router(example_router)

    # Generated routes will be added here by the compiler:
    # from .{entity_name} import router as {entity_name}_router
    # app.include_router({entity_name}_router)
