"""
Example Generated Route Handler

This file demonstrates the structure of generated route handlers.
The compiler will generate similar files for each entity's transitions.

Pattern:
  POST /api/{entity_type}/{event}
  - Extracts Bearer token → FirebaseUser (@user context)
  - Evaluates guards using @user, @entity, @payload
  - Executes effects
  - Returns { data, clientEffects }
  - Broadcasts clientEffects via WebSocket
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from ...core import (
    EventRequest,
    EventResponse,
    EffectExecutor,
    Repository,
    InMemoryRepository,
    FirestoreRepository,
    resolve_binding,
    connection_manager,
)
from ...middleware import FirebaseUser, get_optional_user

# Router for this entity type
router = APIRouter(prefix="/api/tasks", tags=["tasks"])

# Repository instance (configured based on environment)
# In production, use FirestoreRepository()
repository: Repository = InMemoryRepository()


def evaluate_guard(guard_expr: list, context: dict) -> bool:
    """
    Evaluate S-expression guard.

    Guards have access to:
    - @user: Firebase authenticated user (uid, email, role, claims)
    - @entity: Current entity being acted upon
    - @payload: Event payload from client

    Examples:
        (= @user.role "admin")
        (= @entity.owner_id @user.uid)
        (and (= @user.role "editor") (= @entity.status "draft"))
    """
    if not guard_expr:
        return True

    op = guard_expr[0]

    if op == "=":
        left = resolve_binding(guard_expr[1], context)
        right = resolve_binding(guard_expr[2], context)
        return left == right
    elif op == "!=":
        left = resolve_binding(guard_expr[1], context)
        right = resolve_binding(guard_expr[2], context)
        return left != right
    elif op == "and":
        return all(evaluate_guard(g, context) for g in guard_expr[1:])
    elif op == "or":
        return any(evaluate_guard(g, context) for g in guard_expr[1:])
    elif op == "not":
        return not evaluate_guard(guard_expr[1], context)
    elif op == "in":
        value = resolve_binding(guard_expr[1], context)
        collection = resolve_binding(guard_expr[2], context)
        return value in collection if collection else False

    return True


@router.post("/{task_id}/complete", response_model=EventResponse)
async def complete_task(
    task_id: str,
    request: EventRequest,
    background_tasks: BackgroundTasks,
    user: Optional[FirebaseUser] = Depends(get_optional_user),
):
    """
    Handle task:complete event.

    Generated from:
        transition complete {
            from: active
            to: completed
            guard: (or (= @user.role "admin") (= @entity.owner_id @user.uid))
            effects: [
                (set @entity.completed_at @payload.timestamp)
                (persist update task @entity)
                (notify "Task completed!" "success")
            ]
        }
    """
    # Fetch entity for @entity context
    entity = await repository.get("tasks", task_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Task not found")

    # Build evaluation context
    context = {
        "user": user.model_dump() if user else None,
        "entity": entity,
        "payload": request.payload,
        "entityId": task_id,
    }

    # Evaluate guard: (or (= @user.role "admin") (= @entity.owner_id @user.uid))
    guard_expr = [
        "or",
        ["=", "@user.role", "admin"],
        ["=", "@entity.owner_id", "@user.uid"],
    ]

    if not evaluate_guard(guard_expr, context):
        raise HTTPException(
            status_code=403,
            detail="Guard failed: User not authorized to complete this task",
        )

    # Check state transition: from active
    if entity.get("state") != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state transition: expected 'active', got '{entity.get('state')}'",
        )

    # Execute effects
    executor = EffectExecutor(repository)

    effects = [
        ["set", "@entity.completed_at", request.payload.get("timestamp")],
        ["set", "@entity.state", "completed"],
        ["persist", "update", "tasks", "@entity"],
        ["notify", "Task completed!", "success"],
    ]

    for effect in effects:
        await executor.execute(effect, context)

    # Broadcast client effects via WebSocket (non-blocking)
    if executor.client_effects:
        background_tasks.add_task(
            connection_manager.broadcast_client_effects,
            "tasks",
            task_id,
            "complete",
            executor.client_effects,
            executor.data,
        )

    return EventResponse(
        success=True,
        newState="completed",
        data=executor.data,
        clientEffects=executor.client_effects,
        effectResults=executor.effect_results,
    )


@router.post("/create", response_model=EventResponse)
async def create_task(
    request: EventRequest,
    background_tasks: BackgroundTasks,
    user: Optional[FirebaseUser] = Depends(get_optional_user),
):
    """
    Handle task:create event.

    Generated from:
        transition create {
            to: active
            guard: (= @user.authenticated true)
            effects: [
                (set @entity.owner_id @user.uid)
                (set @entity.state "active")
                (persist create task @entity)
            ]
        }
    """
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required to create tasks",
        )

    # Build context (no entity yet for create)
    context = {
        "user": user.model_dump(),
        "entity": {
            **request.payload,
            "owner_id": user.uid,
            "state": "active",
        },
        "payload": request.payload,
    }

    # Execute effects
    executor = EffectExecutor(repository)

    effects = [
        ["set", "@entity.owner_id", "@user.uid"],
        ["set", "@entity.state", "active"],
        ["persist", "create", "tasks", "@entity"],
    ]

    for effect in effects:
        await executor.execute(effect, context)

    # Broadcast client effects via WebSocket (non-blocking)
    if executor.client_effects:
        entity_id = executor.data.get("tasks", {}).get("id", "new")
        background_tasks.add_task(
            connection_manager.broadcast_client_effects,
            "tasks",
            entity_id,
            "create",
            executor.client_effects,
            executor.data,
        )

    return EventResponse(
        success=True,
        newState="active",
        data=executor.data,
        clientEffects=executor.client_effects,
        effectResults=executor.effect_results,
    )
