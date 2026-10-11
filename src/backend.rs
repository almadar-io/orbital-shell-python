//! Python Backend Implementation
//!
//! Generates Python code from Orbital IR for FastAPI backend with PyTorch ML.
//!
//! Output structure:
//! - src/orbital_app/generated/entities/__init__.py - Pydantic entity models
//! - src/orbital_app/generated/traits/{name}.py - Trait state machine classes
//! - src/orbital_app/generated/routes/{name}.py - FastAPI route handlers
//! - tests/ - Pytest test files

use crate::generators::{generate_entities, generate_tests, generate_traits};
use orbital_core::generate::{Backend, GeneratedFile, GeneratedOutput};
use orbital_core::ir::{OirModule, OirTrait};
use orbital_core::str_utils::{to_kebab_case_simple, to_snake_case};

/// Python code generation backend
pub struct PythonBackend {
    /// Generate type hints
    pub type_hints: bool,
    /// Generate docstrings
    pub docstrings: bool,
}

impl Default for PythonBackend {
    fn default() -> Self {
        Self {
            type_hints: true,
            docstrings: true,
        }
    }
}

impl PythonBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate all Python code from an OIR module
    ///
    /// Paths are relative to the shell's generated_dir (src/orbital_app/generated/).
    fn generate_internal(&self, module: &OirModule) -> Vec<GeneratedFile> {
        let mut files = Vec::new();

        // 1. Generate entity models
        files.push(GeneratedFile::new(
            "entities/__init__.py",
            generate_entities(module),
        ));

        // 2. Generate trait state machines
        let trait_files = generate_traits(module);
        for (filename, content) in trait_files {
            files.push(GeneratedFile::new(format!("traits/{}", filename), content));
        }

        // Generate traits __init__.py
        files.push(GeneratedFile::new(
            "traits/__init__.py",
            self.generate_traits_init(module),
        ));

        // 3. Generate API routes
        for trait_def in &module.traits {
            files.push(GeneratedFile::new(
                format!("routes/{}.py", to_snake_case(&trait_def.name)),
                self.generate_trait_routes(trait_def, module),
            ));
        }

        // Generate routes __init__.py
        files.push(GeneratedFile::new(
            "routes/__init__.py",
            self.generate_routes_init(module),
        ));

        // 4. Generate main generated __init__.py
        files.push(GeneratedFile::new(
            "__init__.py",
            self.generate_main_init(module),
        ));

        // 5. Generate helper modules
        files.push(GeneratedFile::new(
            "nn_helpers.py",
            self.generate_nn_helpers(),
        ));

        files.push(GeneratedFile::new(
            "tensor_helpers.py",
            self.generate_tensor_helpers(),
        ));

        files.push(GeneratedFile::new(
            "train_helpers.py",
            self.generate_train_helpers(),
        ));

        files.push(GeneratedFile::new(
            "effect_helpers.py",
            self.generate_effect_helpers(),
        ));

        files.push(GeneratedFile::new(
            "graph_helpers.py",
            self.generate_graph_helpers(),
        ));

        files.push(GeneratedFile::new(
            "data_helpers.py",
            self.generate_data_helpers(),
        ));

        files.push(GeneratedFile::new(
            "contract_helpers.py",
            self.generate_contract_helpers(),
        ));

        files.push(GeneratedFile::new(
            "ml_effect_helpers.py",
            self.generate_ml_effect_helpers(),
        ));

        files.push(GeneratedFile::new(
            "std_helpers.py",
            self.generate_std_helpers(),
        ));

        files.push(GeneratedFile::new(
            "prob_helpers.py",
            self.generate_prob_helpers(),
        ));

        // 6. Generate tests
        let test_files = generate_tests(module);
        for test_file in test_files {
            files.push(GeneratedFile::new(test_file.path, test_file.content));
        }

        files
    }

    /// Generate traits __init__.py
    fn generate_traits_init(&self, module: &OirModule) -> String {
        let mut code = String::new();

        code.push_str(
            "\"\"\"
Trait State Machines

Auto-generated trait implementations.
\"\"\"\n\n",
        );

        for trait_def in &module.traits {
            let module_name = to_snake_case(&trait_def.name);
            code.push_str(&format!(
                "from .{} import {}, {}State\n",
                module_name, trait_def.name, trait_def.name
            ));
        }

        code.push_str("\n__all__ = [\n");
        for trait_def in &module.traits {
            code.push_str(&format!("    \"{}\",\n", trait_def.name));
            code.push_str(&format!("    \"{}State\",\n", trait_def.name));
        }
        code.push_str("]\n");

        code
    }

    /// Generate API routes for a trait
    fn generate_trait_routes(&self, trait_def: &OirTrait, _module: &OirModule) -> String {
        let mut code = String::new();

        let class_name = &trait_def.name;
        let module_name = to_snake_case(class_name);
        let route_prefix = to_kebab_case_simple(class_name);

        code.push_str(&format!(
            "\"\"\"
{} API Routes

FastAPI routes for {} trait events.
Auto-generated by Orbital Compiler.
\"\"\"\n\n",
            class_name, class_name
        ));

        // Imports
        code.push_str("from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks\n");
        code.push_str("from pydantic import BaseModel, Field, ConfigDict\n");
        code.push_str("from typing import Dict, Any, List, Optional\n");
        code.push_str(&format!(
            "from ..traits.{} import {}, {}State\n",
            module_name, class_name, class_name
        ));
        code.push_str("from ...core.event_bus import EventBus, get_event_bus\n");
        code.push_str("from ...core.repository import InMemoryRepository\n");
        code.push_str("from ...core.websocket import connection_manager\n\n");

        // Dependency factory functions
        code.push_str("# Dependency injection\n");
        code.push_str("_repository = InMemoryRepository()\n\n");
        code.push_str("def get_repository():\n");
        code.push_str("    return _repository\n\n");

        // Router - use kebab-case for URL paths
        code.push_str(&format!(
            "router = APIRouter(prefix=\"/{}\", tags=[\"{}\"])\n\n",
            route_prefix, class_name
        ));

        // Collection: need entity_id
        code.push_str("class EventRequest(BaseModel):\n");
        code.push_str("    \"\"\"Event request payload.\"\"\"\n");
        code.push_str("    event: Optional[str] = None  # For unified /events endpoint\n");
        code.push_str("    entity_id: Optional[str] = Field(None, alias=\"entityId\")\n");
        code.push_str("    payload: Dict[str, Any] = {}\n");
        code.push_str("    currentState: Optional[str] = None\n");
        code.push_str("    model_config = ConfigDict(populate_by_name=True)\n\n");

        // Event response — mirrors the TS EventResponse contract from @app/shared
        code.push_str("class EffectResult(BaseModel):\n");
        code.push_str("    \"\"\"Single effect execution result.\"\"\"\n");
        code.push_str("    effect: str\n");
        code.push_str("    action: Optional[str] = None\n");
        code.push_str("    entity_type: Optional[str] = Field(None, alias=\"entityType\")\n");
        code.push_str("    success: bool\n");
        code.push_str("    data: Optional[Any] = None\n");
        code.push_str("    error: Optional[str] = None\n");
        code.push_str("    model_config = ConfigDict(populate_by_name=True)\n\n");
        code.push_str("class EmittedEvent(BaseModel):\n");
        code.push_str("    \"\"\"Event emitted during handler execution.\"\"\"\n");
        code.push_str("    event: str\n");
        code.push_str("    payload: Optional[Dict[str, Any]] = None\n\n");
        code.push_str("class EventResponse(BaseModel):\n");
        code.push_str("    \"\"\"Event response — matches TS EventResponse contract.\"\"\"\n");
        code.push_str("    success: bool\n");
        code.push_str("    newState: Optional[str] = None\n");
        code.push_str("    data: Dict[str, Any] = {}\n");
        code.push_str("    effectResults: List[EffectResult] = []\n");
        code.push_str("    emittedEvents: List[EmittedEvent] = []\n");
        code.push_str("    error: Optional[str] = None\n\n");

        // Trait instance
        code.push_str(&format!(
            "# Trait instance\n_trait = {}()\n\n",
            class_name
        ));

        // Single unified events endpoint — matches TS contract (one POST /events per trait)
        code.push_str("@router.post(\"/events\", response_model=EventResponse)\n");
        code.push_str("async def handle_event(\n");
        code.push_str("    request: EventRequest,\n");
        code.push_str("    background_tasks: BackgroundTasks,\n");
        code.push_str("    repository = Depends(get_repository),\n");
        code.push_str("    event_bus: EventBus = Depends(get_event_bus),\n");
        code.push_str(") -> EventResponse:\n");
        code.push_str("    \"\"\"Handle trait events. Dispatches on the 'event' field.\"\"\"\n");
        code.push_str("    if not request.event:\n");
        code.push_str(
            "        return EventResponse(success=False, error=\"Missing 'event' field\")\n",
        );
        // Validate that the event key is known
        if !trait_def.events.is_empty() {
            let known: Vec<String> = trait_def
                .events
                .iter()
                .map(|e| format!("\"{}\"", e.key))
                .collect();
            code.push_str(&format!(
                "    known_events = {{{}}}\n",
                known.join(", ")
            ));
            code.push_str("    if request.event not in known_events:\n");
            code.push_str("        return EventResponse(success=False, error=f\"Unknown event: {request.event}\")\n");
        }
        code.push_str("    try:\n");

        code.push_str(&format!(
            "        entity = await repository.get(\"{}\", request.entity_id) if request.entity_id else {{}}\n",
            module_name
        ));
        code.push_str("        if not entity:\n");
        code.push_str(
            "            entity = {\"id\": request.entity_id, **(request.payload or {})}\n",
        );

        code.push_str("        # Upsert before dispatch so cross-trait bus handlers loading\n");
        code.push_str("        # this row mid-chain see live @entity.* mutations.\n");
        code.push_str("        if request.entity_id:\n");
        code.push_str(&format!(
            "            await repository.update(\"{}\", request.entity_id, entity)\n",
            module_name
        ));

        code.push_str("        _trait.set_event_bus(event_bus)\n");
        code.push_str("        effect_results, emitted_events = await _trait.dispatch(request.event, entity, request.payload or {})\n");

        code.push_str("        if request.entity_id:\n");
        code.push_str(&format!(
            "            await repository.update(\"{}\", request.entity_id, entity)\n",
            module_name
        ));

        // Broadcast client-bound effects via WebSocket
        code.push_str("        client_effects = [e for e in effect_results if e.get(\"effect\") in ('render-ui', 'navigate', 'notify', 'emit')]\n");
        code.push_str("        if client_effects:\n");
        code.push_str("            background_tasks.add_task(\n");
        code.push_str("                connection_manager.broadcast_client_effects,\n");
        code.push_str(&format!("                \"{}\",\n", module_name));
        code.push_str("                request.entity_id or \"unknown\",\n");
        code.push_str("                request.event,\n");
        code.push_str("                list(client_effects),\n");
        code.push_str("                entity,\n");
        code.push_str("            )\n");
        code.push_str("        return EventResponse(\n");
        code.push_str("            success=True,\n");
        code.push_str("            newState=_trait.state.value,\n");
        code.push_str("            effectResults=effect_results,\n");
        code.push_str("            emittedEvents=emitted_events,\n");
        code.push_str("        )\n");
        code.push_str("    except HTTPException:\n");
        code.push_str("        raise\n");
        code.push_str("    except Exception as e:\n");
        code.push_str("        return EventResponse(success=False, error=str(e))\n\n");

        // State endpoint
        code.push_str("@router.get(\"/state\")\n");
        code.push_str("async def get_state() -> Dict[str, str]:\n");
        code.push_str("    \"\"\"Get current trait state.\"\"\"\n");
        code.push_str("    return {\"state\": _trait.state.value}\n\n");

        // Reset endpoint (for testing)
        let initial_state = trait_def
            .states
            .iter()
            .find(|s| s.is_initial)
            .map(|s| s.name.to_uppercase().replace('-', "_"))
            .unwrap_or_else(|| "IDLE".to_string());
        code.push_str("@router.post(\"/reset\")\n");
        code.push_str("async def reset_state() -> Dict[str, str]:\n");
        code.push_str("    \"\"\"Reset trait to initial state. For testing only.\"\"\"\n");
        code.push_str("    import os\n");
        code.push_str("    if not os.environ.get(\"TESTING\"):\n");
        code.push_str("        raise HTTPException(status_code=403, detail=\"Reset only available in test mode\")\n");
        code.push_str(&format!(
            "    _trait._state = {}State.{}\n",
            class_name, initial_state
        ));
        code.push_str("    # Clear repository data for this trait\n");
        code.push_str(&format!(
            "    _repository._store[\"{}\"] = {{}}\n",
            module_name
        ));
        code.push_str("    # Clear architecture cache\n");
        code.push_str("    _trait._architectures.clear()\n");
        code.push_str("    return {\"state\": _trait.state.value}\n");

        // Set entity endpoint (for testing) - allows setting up entity state for guarded transitions
        code.push_str("\n@router.post(\"/set-entity\")\n");
        code.push_str("async def set_entity_state(request: Dict[str, Any]) -> Dict[str, str]:\n");
        code.push_str(
            "    \"\"\"Set entity fields for testing. Allows satisfying guard conditions.\"\"\"\n",
        );
        code.push_str("    import os\n");
        code.push_str("    if not os.environ.get(\"TESTING\"):\n");
        code.push_str("        raise HTTPException(status_code=403, detail=\"Set-entity only available in test mode\")\n");
        code.push_str("    entity_id = request.get(\"id\", \"test-1\")\n");
        code.push_str(&format!(
            "    entity = _repository._store.get(\"{}\", {{}}).get(entity_id, {{\"id\": entity_id}})\n",
            module_name
        ));
        code.push_str("    for key, value in request.items():\n");
        code.push_str("        entity[key] = value\n");
        code.push_str(&format!(
            "    if \"{}\" not in _repository._store:\n        _repository._store[\"{}\"] = {{}}\n",
            module_name, module_name
        ));
        code.push_str(&format!(
            "    _repository._store[\"{}\"][entity_id] = entity\n",
            module_name
        ));
        code.push_str("    return {\"status\": \"ok\"}\n");

        code
    }

    /// Generate routes __init__.py
    fn generate_routes_init(&self, module: &OirModule) -> String {
        let mut code = String::new();

        code.push_str(
            "\"\"\"
API Routes

Auto-generated FastAPI routes + cross-trait event wiring.
\"\"\"\n\n",
        );

        code.push_str("from fastapi import APIRouter\n\n");

        for trait_def in &module.traits {
            let module_name = to_snake_case(&trait_def.name);
            code.push_str(&format!(
                "from .{} import router as {}_router, _trait as _{}_trait, _repository as _{}_repo\n",
                module_name, module_name, module_name, module_name
            ));
        }

        code.push_str("\n# Combined router\n");
        code.push_str("router = APIRouter()\n\n");

        for trait_def in &module.traits {
            let module_name = to_snake_case(&trait_def.name);
            code.push_str(&format!("router.include_router({}_router)\n", module_name));
        }

        // Build a central registry of trait instances + their repositories so the
        // event-bus wiring can dispatch cross-trait listens. Each entry is
        // (trait_instance, repository, entity_type) keyed by trait class name.
        code.push_str("\n# Central registry: trait_name -> (instance, repository, entity_type)\n");
        code.push_str("TRAIT_REGISTRY = {\n");
        for trait_def in &module.traits {
            let module_name = to_snake_case(&trait_def.name);
            code.push_str(&format!(
                "    \"{}\": (_{}_trait, _{}_repo, \"{}\"),\n",
                trait_def.name, module_name, module_name, module_name
            ));
        }
        code.push_str("}\n");

        // Cross-trait event wiring — mirrors the JS runtime's setupEventListeners().
        // For each trait's LISTENS, subscribe on the shared bus; the handler filters
        // by source trait, projects the payload via `payload_mapping`, forwards
        // entityId, and dispatches `triggers` on the listening trait. Because
        // EventBus.emit awaits its handlers, a dispatch that emits more events
        // recurses naturally — the full event chain drains in one call.
        code.push_str("\n\ndef setup_event_listeners(event_bus):\n");
        code.push_str("    \"\"\"Subscribe every trait's cross-trait listens on the shared bus.\"\"\"\n");
        code.push_str("    for _trait_name, (_inst, _repo, _et) in TRAIT_REGISTRY.items():\n");
        code.push_str("        for _l in _inst.LISTENS:\n");
        code.push_str("            _src_event = _l[\"event\"]\n");
        code.push_str("            _src_trait = _l[\"source_trait\"]\n");
        code.push_str("            _source_any = _l[\"source_any\"]\n");
        code.push_str("            _triggers = _l[\"triggers\"]\n");
        code.push_str("            _mapping = _l[\"payload_mapping\"]\n");
        code.push_str("            _target = _inst\n");
        code.push_str("            _target_repo = _repo\n");
        code.push_str("            _target_et = _et\n");
        code.push_str("            async def _handler(_payload,\n");
        code.push_str("                               _s=_src_trait, _a=_source_any, _tr=_triggers,\n");
        code.push_str("                               _m=_mapping, _t=_target, _r=_target_repo, _e=_target_et):\n");
        code.push_str("                # Source filter: skip unless the emit came from the declared\n");
        code.push_str("                # source trait (or source_any accepts everything).\n");
        code.push_str("                if not _a and _s and _payload.get(\"_source_trait\") != _s:\n");
        code.push_str("                    return\n");
        code.push_str("                # Strip internal source/entity tags before forwarding.\n");
        code.push_str("                _p = {k: v for k, v in _payload.items() if k not in (\"_source_trait\", \"_entity_id\")}\n");
        code.push_str("                # Apply payload projection (with { ... }) if declared.\n");
        code.push_str("                if _m is not None:\n");
        code.push_str("                    _p = {k: fn(_p) for k, fn in _m.items()}\n");
        code.push_str("                # Load the entity by forwarded id (or any existing row) so\n");
        code.push_str("                # @entity.* binds against persisted state across the chain.\n");
        code.push_str("                _eid = _payload.get(\"_entity_id\") or _p.get(\"entityId\") or _p.get(\"orbitalName\")\n");
        code.push_str("                _entity = await _r.get(_e, _eid) if _eid else (await _r.first(_e))\n");
        code.push_str("                if not _entity:\n");
        code.push_str("                    _entity = {\"id\": _eid, **_p} if _eid else dict(_p)\n");
        code.push_str("                _t.set_event_bus(event_bus)\n");
        code.push_str("                await _t.dispatch(_tr, _entity, _p)\n");
        code.push_str("                # Persist so the next hop sees mutated fields.\n");
        code.push_str("                await _r.update(_e, _entity.get(\"id\"), _entity)\n");
        code.push_str("            event_bus.subscribe(_src_event, _handler)\n");

        code.push_str("\n\ndef register_routes(app):\n");
        code.push_str("    \"\"\"Register all generated routes with the FastAPI app.\"\"\"\n");
        code.push_str("    app.include_router(router, prefix=\"/api\")\n");
        code.push_str("    # Wire cross-trait listens on the shared bus.\n");
        code.push_str("    from ...core.event_bus import get_event_bus\n");
        code.push_str("    setup_event_listeners(get_event_bus())\n");
        // Wire the runtime sidecar as the call-service client on every trait
        // instance, so `call-service "<svc>" "<action>"` effects reach the
        // JS runtime's real integrations (@almadar/llm, …) over /api/eval.
        // Stateless: no schema upload — /api/eval dispatches integrations
        // directly. If SIDECAR_URL is unset the client falls back to the
        // default localhost URL; a call-service then fails clearly, routing
        // to the trait's failure event (e.g. EMBED_FAILED).
        code.push_str("    from ...core.runtime_client import RuntimeSidecar\n");
        code.push_str("    _sidecar = RuntimeSidecar()\n");
        code.push_str("    for _name, (_inst, _repo, _et) in TRAIT_REGISTRY.items():\n");
        code.push_str("        _inst.set_service_client(_sidecar.call_service)\n");

        code
    }

    /// Generate main __init__.py
    fn generate_main_init(&self, _module: &OirModule) -> String {
        let mut code = String::new();

        code.push_str(
            "\"\"\"
Generated Orbital Code

Auto-generated by Orbital Compiler.
Do not edit manually.
\"\"\"\n\n",
        );

        code.push_str("from . import entities\n");
        code.push_str("from . import traits\n");
        code.push_str("from . import routes\n");
        code.push_str("from .nn_helpers import *\n");
        code.push_str("from .tensor_helpers import *\n");
        code.push_str("from .train_helpers import *\n");
        code.push_str("from .effect_helpers import *\n");
        code.push_str("from .graph_helpers import *\n");
        code.push_str("from .data_helpers import *\n");
        code.push_str("from .contract_helpers import *\n");
        code.push_str("from .ml_effect_helpers import *\n");
        code.push_str("from .std_helpers import *\n");
        code.push_str("from .prob_helpers import *\n\n");

        code.push_str("__all__ = [\n");
        code.push_str("    \"entities\",\n");
        code.push_str("    \"traits\",\n");
        code.push_str("    \"routes\",\n");
        code.push_str("]\n");

        code
    }

    /// Generate NN helper functions
    fn generate_nn_helpers(&self) -> String {
        r#""""
Neural Network Helpers

Helper functions for nn/* S-expressions.
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Any


def set_weights(module: nn.Module, weights: torch.Tensor) -> None:
    """Set network weights from a flat tensor."""
    offset = 0
    with torch.no_grad():
        for param in module.parameters():
            numel = param.numel()
            param.copy_(weights[offset:offset + numel].view(param.shape))
            offset += numel


def build_network(architecture: list) -> nn.Module:
    """Build a PyTorch module from S-expression architecture."""
    if not architecture:
        return nn.Identity()

    op = architecture[0]

    if op == "nn/sequential":
        layers = [build_layer(layer) for layer in architecture[1:]]
        return nn.Sequential(*layers)

    return build_layer(architecture)


def build_layer(layer: list) -> nn.Module:
    """Build a single layer from S-expression."""
    if not layer:
        return nn.Identity()

    op = layer[0]

    if op == "nn/linear":
        in_features = layer[1] if len(layer) > 1 else 1
        out_features = layer[2] if len(layer) > 2 else 1
        return nn.Linear(in_features, out_features)
    elif op == "nn/relu":
        return nn.ReLU()
    elif op == "nn/tanh":
        return nn.Tanh()
    elif op == "nn/sigmoid":
        return nn.Sigmoid()
    elif op == "nn/softmax":
        dim = layer[1] if len(layer) > 1 else -1
        return nn.Softmax(dim=dim)
    elif op == "nn/dropout":
        p = layer[1] if len(layer) > 1 else 0.5
        return nn.Dropout(p=p)
    elif op == "nn/batchnorm":
        num_features = layer[1] if len(layer) > 1 else 1
        return nn.BatchNorm1d(num_features)
    elif op == "nn/layernorm":
        normalized_shape = layer[1] if len(layer) > 1 else 1
        return nn.LayerNorm(normalized_shape)

    raise ValueError(f"Unknown layer type: {op}")


def clone_module(module: nn.Module) -> nn.Module:
    """Deep-copy a module, weights included."""
    return copy.deepcopy(module)
"#
        .to_string()
    }

    /// Generate tensor helper functions
    fn generate_tensor_helpers(&self) -> String {
        r#""""
Tensor Helpers

Helper functions for tensor/* S-expressions.
"""

import torch
from typing import Dict, Any, List


def clamp_per_dim(tensor: torch.Tensor, ranges: Dict[str, Dict[str, float]]) -> torch.Tensor:
    """Clamp each dimension to its specified range."""
    result = tensor.clone()
    for dim_str, bounds in ranges.items():
        dim = int(dim_str)
        if dim < len(result):
            min_val = bounds.get("min", float("-inf"))
            max_val = bounds.get("max", float("inf"))
            result[dim] = result[dim].clamp(min_val, max_val)
    return result


def out_of_range_dims(tensor: torch.Tensor, ranges: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Get dimensions that exceed their specified ranges."""
    violations = []
    for dim_str, bounds in ranges.items():
        dim = int(dim_str)
        if dim < len(tensor):
            value = tensor[dim].item()
            min_val = bounds.get("min", float("-inf"))
            max_val = bounds.get("max", float("inf"))

            if value < min_val or value > max_val:
                violations.append({
                    "dim": dim,
                    "value": value,
                    "min": min_val,
                    "max": max_val,
                    "meaning": bounds.get("meaning", f"dim_{dim}"),
                })

    return violations


def out_of_range_indices(tensor: torch.Tensor, ranges: Dict[str, Dict[str, float]]) -> List[int]:
    """Get the flat indices of tensor elements outside their specified range."""
    indices = []
    flat = tensor.flatten()
    for dim_str, bounds in ranges.items():
        idx = int(dim_str)
        if idx < len(flat):
            value = flat[idx].item()
            min_val = bounds.get("min", float("-inf"))
            max_val = bounds.get("max", float("inf"))
            if value < min_val or value > max_val:
                indices.append(idx)
    return indices
"#.to_string()
    }

    /// Generate training helper functions
    fn generate_train_helpers(&self) -> String {
        r#""""
Training Helpers

Helper functions for train/* S-expressions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional


def train_loop(
    module: nn.Module,
    data: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute training loop with constraints. Returns dict with finalLoss.

    Data can be:
    - List of dicts with "observation" and "target" keys (supervised)
    - List of arrays (self-supervised, input = target truncated to output size)
    - Single tensor/array (batch training)
    """
    epochs = config.get("epochs", 100)
    lr = config.get("learningRate", 0.001)
    max_grad_norm = config.get("maxGradientNorm")
    max_weight_mag = config.get("maxWeightMagnitude")
    forbidden = config.get("forbiddenOutputRegions", [])

    # Use provided optimizer or create default
    optimizer = config.get("optimizer")
    if optimizer is None:
        optimizer = optim.Adam(module.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    constraint_violations = []

    # Normalize data format
    if isinstance(data, torch.Tensor):
        data = [{"observation": row.tolist(), "target": row.tolist()} for row in data]
    elif isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], (list, tuple)):
            # List of arrays - self-supervised
            data = [{"observation": d, "target": d} for d in data]
        elif isinstance(data[0], dict) and "observation" in data[0]:
            pass  # Already in correct format
        else:
            # Fallback: treat as single sample
            data = [{"observation": data, "target": data}]

    for epoch in range(epochs):
        total_loss = 0.0
        violations = 0

        for sample in data:
            optimizer.zero_grad()

            # Forward pass
            obs = sample.get("observation", sample.get("input", []))
            tgt = sample.get("target", sample.get("output", obs))
            input_tensor = torch.tensor(obs, dtype=torch.float32)
            output = module(input_tensor)
            # Truncate target to match output size
            target = torch.tensor(tgt[:len(output)] if len(tgt) > len(output) else tgt, dtype=torch.float32)

            # Base loss
            loss = criterion(output, target)

            # Constraint loss
            for region in forbidden:
                dim = region["dim"]
                min_val = region["min"]
                max_val = region["max"]
                value = output[dim]

                if min_val <= value.item() <= max_val:
                    dist_to_min = (value - min_val).abs()
                    dist_to_max = (max_val - value).abs()
                    loss = loss + 10.0 * torch.min(dist_to_min, dist_to_max)
                    violations += 1

            # Backward pass
            loss.backward()

            # Gradient clipping
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm)

            # Update weights
            optimizer.step()

            # Weight clipping
            if max_weight_mag:
                clip_weights(module, max_weight_mag)

            total_loss += loss.item()

        avg_loss = total_loss / len(data) if data else 0.0
        losses.append(avg_loss)
        constraint_violations.append(violations)

    return {
        "finalLoss": losses[-1] if losses else 0.0,
        "losses": losses,
        "constraintViolations": constraint_violations,
        "epochs": epochs,
    }


def train_step(
    module: nn.Module,
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute single training step."""
    lr = config.get("learningRate", 0.001)
    optimizer = optim.Adam(module.parameters(), lr=lr)

    optimizer.zero_grad()
    output = module(input_tensor)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "output": output.detach()}


def validate(
    module: nn.Module,
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate model on test cases."""
    module.eval()
    total_loss = 0.0
    passed = 0

    with torch.no_grad():
        for case in test_cases:
            input_tensor = torch.tensor(case["input"], dtype=torch.float32)
            expected = torch.tensor(case["expected"], dtype=torch.float32)
            tolerance = case.get("tolerance", 0.1)

            output = module(input_tensor)
            loss = F.mse_loss(output, expected).item()
            total_loss += loss

            if loss < tolerance:
                passed += 1

    return {
        "total": len(test_cases),
        "passed": passed,
        "failed": len(test_cases) - passed,
        "avg_loss": total_loss / len(test_cases) if test_cases else 0,
    }


def check_constraints(
    module: nn.Module,
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """Check if network weights satisfy all constraints."""
    violations = []

    max_magnitude = constraints.get("maxWeightMagnitude")
    if max_magnitude:
        for name, param in module.named_parameters():
            max_val = param.abs().max().item()
            if max_val > max_magnitude:
                violations.append({
                    "type": "weight_magnitude",
                    "param": name,
                    "max_allowed": max_magnitude,
                    "actual": max_val,
                })

    return {
        "satisfied": len(violations) == 0,
        "violations": violations,
    }


def clip_weights(module: nn.Module, max_magnitude: float) -> Dict[str, Any]:
    """Clip weights to max magnitude. Returns effect dict."""
    with torch.no_grad():
        for param in module.parameters():
            param.clamp_(-max_magnitude, max_magnitude)
    return {"type": "clip-weights", "maxMagnitude": max_magnitude}


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns from rewards."""
    returns = []
    running_return = 0

    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)

    return torch.tensor(returns, dtype=torch.float32)


def compute_advantages(
    rewards: List[float],
    values: torch.Tensor,
    config: Dict[str, Any],
) -> torch.Tensor:
    """Compute GAE advantages."""
    gamma = config.get("gamma", 0.99)
    lam = config.get("lambda", 0.95)

    advantages = []
    gae = 0

    for i in reversed(range(len(rewards))):
        next_value = values[i + 1] if i + 1 < len(values) else 0
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float32)


def check_weight_magnitude(module: nn.Module, max_magnitude: float) -> bool:
    """Check if all weights are within magnitude bounds."""
    with torch.no_grad():
        for param in module.parameters():
            if param.abs().max().item() > max_magnitude:
                return False
    return True


def check_forbidden_outputs(
    module: nn.Module,
    forbidden_regions: List[Dict[str, Any]],
    test_inputs: Optional[List[torch.Tensor]] = None,
) -> bool:
    """Check if model outputs avoid forbidden regions.

    Args:
        module: The neural network module to check
        forbidden_regions: List of dicts with 'dim', 'min', 'max' keys
        test_inputs: Optional list of test inputs (generates random if None)

    Returns:
        True if all outputs are outside forbidden regions
    """
    module.eval()

    if test_inputs is None:
        # Generate random test inputs
        # Try to infer input size from first layer
        for param in module.parameters():
            in_features = param.shape[-1] if len(param.shape) > 1 else param.shape[0]
            break
        test_inputs = [torch.randn(in_features) for _ in range(10)]

    with torch.no_grad():
        for input_tensor in test_inputs:
            output = module(input_tensor)
            for region in forbidden_regions:
                dim = region.get("dim", 0)
                min_val = region.get("min", float("-inf"))
                max_val = region.get("max", float("inf"))

                if dim < len(output):
                    value = output[dim].item()
                    if min_val <= value <= max_val:
                        return False

    return True


def check_regression(baseline: float, current: float, tolerance: float = 0.0) -> bool:
    """Return True if `current` has not regressed beyond `baseline` (+ tolerance)."""
    return current <= baseline + tolerance


def get_gradient_norm(module: nn.Module) -> float:
    """Return the total L2 norm of all parameter gradients."""
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.data.norm(2).item() ** 2)
    return total ** 0.5


def get_max_weight_magnitude(module: nn.Module) -> float:
    """Return the maximum absolute weight value across all parameters."""
    max_mag = 0.0
    with torch.no_grad():
        for param in module.parameters():
            m = param.abs().max().item()
            if m > max_mag:
                max_mag = m
    return max_mag


def sample_batch(data: List[Any], batch_size: int) -> List[Any]:
    """Randomly sample a mini-batch of `batch_size` items from `data`."""
    import random
    if not data:
        return []
    n = min(int(batch_size), len(data))
    return random.sample(list(data), n)
"#.to_string()
    }

    /// Generate effect helper functions
    fn generate_effect_helpers(&self) -> String {
        r#""""
Effect Helpers

Helper functions for Orbital effect S-expressions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class NotificationSeverity(Enum):
    """Notification severity levels."""
    INFO = "Info"
    WARNING = "Warning"
    ERROR = "Error"
    SUCCESS = "Success"


@dataclass
class Effect:
    """Base effect class."""
    effect_type: str
    data: Dict[str, Any]


def notify(message: str, severity: str = "Info") -> Dict[str, Any]:
    """Create a notification effect."""
    return {
        "type": "notify",
        "message": message,
        "severity": severity,
    }


def render_ui(slot: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a UI render effect."""
    return {
        "type": "render-ui",
        "slot": slot,
        "data": data,
    }


def navigate(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a navigation effect."""
    return {
        "type": "navigate",
        "path": path,
        "params": params or {},
    }


def spawn(entity_type: str, props: Dict[str, Any]) -> Dict[str, Any]:
    """Create a spawn entity effect."""
    return {
        "type": "spawn",
        "entity_type": entity_type,
        "props": props,
    }


def despawn(entity_id: str) -> Dict[str, Any]:
    """Create a despawn entity effect."""
    return {
        "type": "despawn",
        "entity_id": entity_id,
    }


def emit(event: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create an emit event effect."""
    return {
        "type": "emit",
        "event": event,
        "data": data or {},
    }


def custom_effect(name: str, **kwargs) -> Dict[str, Any]:
    """Create a custom effect."""
    return {
        "type": "custom",
        "name": name,
        "args": kwargs,
    }


def now() -> float:
    """Get current timestamp in milliseconds."""
    import time
    return time.time() * 1000
"#
        .to_string()
    }

    /// Generate graph helper functions
    fn generate_graph_helpers(&self) -> String {
        r#""""
Graph Helpers

Helper functions for graph/* S-expressions.
"""

import torch
from typing import Dict, Any, List, Optional


class GraphData:
    """Simple graph representation for GNN operations."""

    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor,
                 edge_attr: Optional[torch.Tensor] = None):
        self.x = x  # Node features [num_nodes, feature_dim]
        self.edge_index = edge_index  # Edge indices [2, num_edges]
        self.edge_attr = edge_attr  # Edge features [num_edges, edge_dim]
        self.num_nodes = x.size(0)

    def clone(self) -> "GraphData":
        return GraphData(
            x=self.x.clone(),
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone() if self.edge_attr is not None else None,
        )


def graph_from_entities(entities: List[Dict[str, Any]]) -> GraphData:
    """Build a graph from entity dicts. Each entity becomes a node."""
    if not entities:
        return GraphData(
            x=torch.zeros(0, 1),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )
    # Extract numeric fields as features
    features = []
    for e in entities:
        row = [float(v) for v in e.values() if isinstance(v, (int, float))]
        features.append(row if row else [0.0])
    max_len = max(len(r) for r in features)
    features = [r + [0.0] * (max_len - len(r)) for r in features]
    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    return GraphData(x=x, edge_index=edge_index)


def graph_from_adjacency(features: Any, adj_matrix: Any) -> GraphData:
    """Build a graph from a feature matrix and adjacency matrix."""
    x = torch.tensor(features, dtype=torch.float32) if not isinstance(features, torch.Tensor) else features
    adj = torch.tensor(adj_matrix, dtype=torch.float32) if not isinstance(adj_matrix, torch.Tensor) else adj_matrix
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return GraphData(x=x, edge_index=edge_index)


def graph_from_edge_list(features: Any, edge_list: Any) -> GraphData:
    """Build a graph from node features and an edge list [[src, dst], ...]."""
    x = torch.tensor(features, dtype=torch.float32) if not isinstance(features, torch.Tensor) else features
    edges = torch.tensor(edge_list, dtype=torch.long) if not isinstance(edge_list, torch.Tensor) else edge_list
    if edges.dim() == 2 and edges.size(1) == 2:
        edge_index = edges.t().contiguous()
    else:
        edge_index = edges
    return GraphData(x=x, edge_index=edge_index)


def graph_add_self_loops(graph: GraphData) -> GraphData:
    """Add self-loop edges to every node."""
    g = graph.clone()
    n = g.num_nodes
    self_loops = torch.arange(n, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    g.edge_index = torch.cat([g.edge_index, self_loops], dim=1)
    return g


def graph_to_undirected(graph: GraphData) -> GraphData:
    """Make graph undirected by adding reverse edges."""
    g = graph.clone()
    reverse = g.edge_index.flip(0)
    g.edge_index = torch.cat([g.edge_index, reverse], dim=1)
    return g


def graph_subgraph(graph: GraphData, node_mask: Any) -> GraphData:
    """Extract a subgraph using a boolean node mask."""
    mask = torch.tensor(node_mask, dtype=torch.bool) if not isinstance(node_mask, torch.Tensor) else node_mask
    x = graph.x[mask]
    # Remap edge indices
    node_map = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    node_map[mask] = torch.arange(mask.sum(), dtype=torch.long)
    src, dst = graph.edge_index
    keep = mask[src] & mask[dst]
    new_src = node_map[src[keep]]
    new_dst = node_map[dst[keep]]
    edge_index = torch.stack([new_src, new_dst])
    return GraphData(x=x, edge_index=edge_index)


def graph_k_hop(graph: GraphData, node_idx: int, k: int) -> GraphData:
    """Extract k-hop subgraph around a node."""
    visited = {node_idx}
    frontier = {node_idx}
    src, dst = graph.edge_index.tolist()
    for _ in range(k):
        next_frontier = set()
        for n in frontier:
            for s, d in zip(src, dst):
                if s == n and d not in visited:
                    next_frontier.add(d)
                if d == n and s not in visited:
                    next_frontier.add(s)
        visited |= next_frontier
        frontier = next_frontier
    mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    for n in visited:
        mask[n] = True
    return graph_subgraph(graph, mask)


def graph_node_features(graph: GraphData) -> torch.Tensor:
    """Get node feature matrix."""
    return graph.x


def graph_edge_index(graph: GraphData) -> torch.Tensor:
    """Get edge index tensor."""
    return graph.edge_index


def graph_edge_features(graph: GraphData) -> Optional[torch.Tensor]:
    """Get edge feature matrix (or None)."""
    return graph.edge_attr


def graph_num_nodes(graph: GraphData) -> int:
    """Get number of nodes."""
    return graph.num_nodes


def graph_num_edges(graph: GraphData) -> int:
    """Get number of edges."""
    return graph.edge_index.size(1)


def graph_degree(graph: GraphData) -> torch.Tensor:
    """Get degree of each node."""
    deg = torch.zeros(graph.num_nodes, dtype=torch.long)
    src = graph.edge_index[0]
    for s in src.tolist():
        deg[s] += 1
    return deg


def graph_batch(graphs: List[GraphData]) -> GraphData:
    """Batch multiple graphs into one with offset edge indices."""
    if not graphs:
        return GraphData(
            x=torch.zeros(0, 1),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )
    xs, edge_indices = [], []
    offset = 0
    for g in graphs:
        xs.append(g.x)
        edge_indices.append(g.edge_index + offset)
        offset += g.num_nodes
    return GraphData(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
    )
"#.to_string()
    }

    /// Generate data preprocessing helper functions
    fn generate_data_helpers(&self) -> String {
        r#""""
Data Helpers

Helper functions for data/* S-expressions.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Any, List, Tuple, Optional


class OrbitalDataset(Dataset):
    """Simple dataset wrapping lists of dicts or tensors."""

    def __init__(self, data: List[Any], config: Dict[str, Any]):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item, dict):
            x = torch.tensor(item.get("input", item.get("observation", [])), dtype=torch.float32)
            y = torch.tensor(item.get("target", item.get("output", [])), dtype=torch.float32)
            return x, y
        if isinstance(item, (list, tuple)):
            return torch.tensor(item, dtype=torch.float32), torch.tensor(item, dtype=torch.float32)
        return item, item


def data_dataset(data: Any, config: Dict[str, Any]) -> OrbitalDataset:
    """Create an OrbitalDataset from raw data and config."""
    if isinstance(data, list):
        return OrbitalDataset(data, config)
    return OrbitalDataset([], config)


def data_dataloader(dataset: OrbitalDataset, config: Dict[str, Any]) -> DataLoader:
    """Create a DataLoader from a dataset and config."""
    batch_size = config.get("batchSize", config.get("batch_size", 32))
    shuffle = config.get("shuffle", True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def data_split(dataset: OrbitalDataset, config: Dict[str, Any]) -> Tuple[OrbitalDataset, OrbitalDataset]:
    """Split a dataset into train/test subsets."""
    ratio = config.get("trainRatio", config.get("train_ratio", 0.8))
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def data_normalize(data: Any, config: Dict[str, Any]) -> torch.Tensor:
    """Normalize data (z-score by default)."""
    t = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
    method = config.get("method", "zscore")
    if method == "zscore":
        mean = t.mean(dim=0, keepdim=True)
        std = t.std(dim=0, keepdim=True).clamp(min=1e-8)
        return (t - mean) / std
    elif method == "minmax":
        mn = t.min(dim=0, keepdim=True).values
        mx = t.max(dim=0, keepdim=True).values
        return (t - mn) / (mx - mn).clamp(min=1e-8)
    return t


def data_augment(data: Any, config: Dict[str, Any]) -> torch.Tensor:
    """Apply simple data augmentation (noise injection)."""
    t = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
    noise_scale = config.get("noiseScale", config.get("noise_scale", 0.01))
    return t + torch.randn_like(t) * noise_scale


def data_tokenize(text: Any, config: Dict[str, Any]) -> List[int]:
    """Simple character/word tokenization."""
    mode = config.get("mode", "char")
    text_str = str(text)
    if mode == "word":
        tokens = text_str.split()
        vocab = config.get("vocab", {})
        return [vocab.get(w, 0) for w in tokens]
    return [ord(c) for c in text_str]


def data_pad(data: Any, config: Dict[str, Any]) -> torch.Tensor:
    """Pad sequences to target length."""
    t = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
    target_len = config.get("length", config.get("targetLength", t.size(-1)))
    pad_value = config.get("padValue", config.get("pad_value", 0.0))
    if t.size(-1) < target_len:
        padding = torch.full((*t.shape[:-1], target_len - t.size(-1)), pad_value)
        return torch.cat([t, padding], dim=-1)
    return t[..., :target_len]
"#.to_string()
    }

    /// Generate contract helper functions
    fn generate_contract_helpers(&self) -> String {
        r#""""
Contract Helpers — pure-Python, mirrors @almadar/core ml-contract.ts.

A tensor here is a plain JSON value (number or nested lists of numbers),
NOT a torch.Tensor — exactly what a model output round-trips to and what
the JS runtime path validates. Same contract JSON in, same {valid, violations} out.
"""

from typing import Dict, Any, List, Union

TensorValue = Union[float, List[Any]]


def _is_tensor(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return isinstance(v, list) and all(_is_tensor(x) for x in v)


def _describe(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, list):
        return "array containing non-numeric values"
    return type(v).__name__


def _shape(t: TensorValue) -> List[int]:
    if isinstance(t, (int, float)):
        return []
    if len(t) == 0:
        return [0]
    first = t[0]
    if isinstance(first, (int, float)):
        return [len(t)]
    return [len(t)] + _shape(first)


def _last_dim_size(t: TensorValue) -> int:
    s = _shape(t)
    return s[-1] if s else 0


def _gather_last_dim(t: TensorValue, dim: int) -> List[float]:
    if isinstance(t, (int, float)):
        return []
    if len(t) == 0:
        return []
    if isinstance(t[0], (int, float)):
        return [float(t[dim])]
    out: List[float] = []
    for sub in t:
        out.extend(_gather_last_dim(sub, dim))
    return out


def _map_last_dim(t: TensorValue, dim: int, fn) -> TensorValue:
    if isinstance(t, (int, float)):
        return t
    if len(t) == 0:
        return t
    if isinstance(t[0], (int, float)):
        copy = list(t)
        copy[dim] = fn(copy[dim])
        return copy
    return [_map_last_dim(sub, dim, fn) for sub in t]


def _validate(tensor: Any, contract: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_tensor(tensor):
        return {"valid": False, "violations": [{"type": "not_a_tensor", "actualType": _describe(tensor)}]}
    violations: List[Dict[str, Any]] = []
    shape = _shape(tensor)
    expected = contract.get("shape")
    if expected is not None and shape != list(expected):
        violations.append({"type": "shape_mismatch", "expected": list(expected), "actual": shape})
    ranges = contract.get("ranges") or {}
    size = _last_dim_size(tensor)
    for dim_str, bounds in ranges.items():
        dim = int(dim_str)
        if dim >= size:
            continue
        vals = _gather_last_dim(tensor, dim)
        mn = bounds.get("min", float("-inf"))
        mx = bounds.get("max", float("inf"))
        amn, amx = min(vals), max(vals)
        if amn < mn or amx > mx:
            violations.append({"type": "range_violation", "dim": dim, "min": mn, "max": mx, "actualMin": amn, "actualMax": amx})
    return {"valid": len(violations) == 0, "violations": violations}


def contract_validate_input(tensor: Any, contract: Dict[str, Any]) -> Dict[str, Any]:
    return _validate(tensor, contract)


def contract_validate_output(tensor: Any, contract: Dict[str, Any]) -> Dict[str, Any]:
    return _validate(tensor, contract)


def contract_clamp_output(tensor: Any, contract: Dict[str, Any]) -> TensorValue:
    result = tensor
    ranges = contract.get("ranges") or {}
    size = _last_dim_size(result)
    for dim_str, bounds in ranges.items():
        dim = int(dim_str)
        if dim >= size:
            continue
        mn = bounds.get("min", float("-inf"))
        mx = bounds.get("max", float("inf"))
        result = _map_last_dim(result, dim, lambda v: max(mn, min(mx, v)))
    return result


def contract_violations(tensor: Any, contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _validate(tensor, contract).get("violations", [])


def contract_entity_to_tensor(entity: Dict[str, Any], mapping: Dict[str, Any]) -> List[float]:
    fields = mapping.get("fields", [])
    out: List[float] = []
    for field in fields:
        name = field if isinstance(field, str) else field.get("name", "")
        val = entity.get(name, 0.0)
        out.append(float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.0)
    return out


def contract_tensor_to_payload(tensor: TensorValue, mapping: Dict[str, Any]) -> Dict[str, Any]:
    fields = mapping.get("fields", [])
    values = tensor if isinstance(tensor, list) else [tensor]
    result: Dict[str, Any] = {}
    for i, field in enumerate(fields):
        name = field if isinstance(field, str) else field.get("name", "")
        if i < len(values):
            result[name] = values[i]
    return result
"#
        .to_string()
    }

    /// Generate ML effect helper functions (forward, train, evaluate, checkpoint)
    fn generate_ml_effect_helpers(&self) -> String {
        r#""""ML Effect Helpers - Forward, Train, Evaluate implementations."""
import os
import torch
import torch.nn as nn
from .nn_helpers import build_network
from .train_helpers import train_loop
from .contract_helpers import contract_violations, contract_clamp_output
from ..core.settings import get_settings


class _VerifyState:
    model = None
    last_output = None
    architecture_spec = None
    training_status = None
    constraint_config = None


verify_state = _VerifyState()


def ml_forward(slot, config):
    """Run forward pass through architecture tree."""
    arch = config.get("architecture")
    input_data = config.get("input")
    on_complete = config.get("on-complete", "PREDICTION_READY")

    if arch is None:
        return {"error": "No architecture provided"}

    # Build model - architecture may be a list (JSON tree) or already an nn.Module
    if isinstance(arch, nn.Module):
        model = arch
    elif isinstance(arch, (list, dict)):
        model = build_network(arch)
    else:
        return {"error": f"Invalid architecture type: {type(arch)}"}
    model.eval()

    # Prepare input
    if isinstance(input_data, list):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
    elif isinstance(input_data, torch.Tensor):
        input_tensor = input_data.float()
    else:
        input_tensor = torch.zeros(1)

    # Ensure batch dimension
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)

    # Input contract validation (soft enforcement: log, don't block)
    input_contract = config.get("input-contract")
    if input_contract and isinstance(input_contract, dict):
        violations = contract_violations(input_tensor, input_contract)
        if violations:
            print(f"WARNING: input contract violations: {violations}")

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Output contract clamping (hard enforcement: clamp into range)
    output_contract = config.get("output-contract")
    if output_contract and isinstance(output_contract, dict):
        output = contract_clamp_output(output, output_contract)

    # Wire verify state
    verify_state.model = model
    verify_state.last_output = output
    verify_state.architecture_spec = arch

    return {"output": output.detach().tolist(), "shape": list(output.shape), "_on_complete": on_complete}


def ml_train(config):
    """Run training loop."""
    arch = config.get("architecture")
    dataset = config.get("dataset")
    train_config = config.get("config", {})
    on_complete = config.get("on-complete", "TRAINING_DONE")

    if arch is None:
        return {"error": "No architecture provided"}

    if isinstance(arch, nn.Module):
        model = arch
    elif isinstance(arch, (list, dict)):
        model = build_network(arch)
    else:
        return {"error": f"Invalid architecture type: {type(arch)}"}

    # Run training
    model, result = train_loop(model, dataset, train_config)

    # Wire verify state
    verify_state.model = model
    verify_state.architecture_spec = arch
    verify_state.training_status = {
        "initial_loss": result.get("initialLoss"),
        "final_loss": result.get("finalLoss"),
        "grad_norm": None,
        "weights_changed": result.get("weightsChanged", True),
        "epochs_completed": result.get("epochsCompleted", 0),
    }

    # Populate constraint config from train_config
    constraints = train_config.get("constraints", {})
    verify_state.constraint_config = constraints

    return {
        "finalLoss": result.get("finalLoss"),
        "initialLoss": result.get("initialLoss"),
        "epochsCompleted": result.get("epochsCompleted"),
        "weights": "trained",
        "history": result.get("history"),
        "_on_complete": on_complete,
    }


def ml_evaluate(config):
    """Run model evaluation."""
    arch = config.get("architecture")
    dataset = config.get("dataset")
    metrics_list = config.get("metrics", ["loss"])

    if arch is None:
        return {"error": "No architecture provided"}

    model = build_network(arch)
    model.eval()

    results = {}
    total_loss = 0.0
    count = 0

    criterion = nn.MSELoss()

    if isinstance(dataset, list):
        with torch.no_grad():
            for item in dataset:
                obs = item.get("observation", item.get("input", []))
                tgt = item.get("target", item.get("label", []))
                inp = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) if not isinstance(obs, torch.Tensor) else obs.float()
                tar = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0) if not isinstance(tgt, torch.Tensor) else tgt.float()
                out = model(inp)
                loss = criterion(out, tar)
                total_loss += loss.item()
                count += 1

    avg_loss = total_loss / max(count, 1)
    results["loss"] = avg_loss

    if "accuracy" in metrics_list:
        results["accuracy"] = 0.0  # Placeholder
    if "f1" in metrics_list:
        results["f1"] = 0.0  # Placeholder

    verify_state.model = model

    return {"metrics": results, "samples": count}


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve a checkpoint path against MODEL_CHECKPOINT_URI when relative.

    Absolute paths are used as-is. Relative paths are joined onto the
    configured checkpoint URI's base directory — only file:// is
    implemented; any other scheme is a loud, unrecoverable error rather
    than a silent local-path fallback.
    """
    if os.path.isabs(path):
        return path

    uri = get_settings().model_checkpoint_uri
    scheme, sep, rest = uri.partition("://")
    if not sep:
        base_dir = uri
    elif scheme == "file":
        base_dir = rest
    else:
        raise ValueError(
            f"Unsupported MODEL_CHECKPOINT_URI scheme '{scheme}://' — only 'file://' is implemented. "
            "Point MODEL_CHECKPOINT_URI at a mounted volume (file:///mnt/checkpoints) or pass an absolute path."
        )
    return os.path.join(base_dir, path)


def checkpoint_save(path, weights):
    """Save model checkpoint."""
    resolved = _resolve_checkpoint_path(path)
    checkpoint_dir = os.path.dirname(resolved)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if isinstance(weights, nn.Module):
        torch.save(weights.state_dict(), resolved)
    elif isinstance(weights, dict):
        torch.save(weights, resolved)
    return {"saved": True, "path": resolved}


def checkpoint_load(path):
    """Load model checkpoint."""
    try:
        resolved = _resolve_checkpoint_path(path)
        state = torch.load(resolved, map_location="cpu", weights_only=True)
        return {"loaded": True, "path": resolved}
    except Exception as e:
        return {"loaded": False, "error": str(e)}
"#.to_string()
    }

    /// Ordinary std-operator helpers (array/nearest, object/get). Mirrors the
    /// per-family helper-module pattern (nn_helpers, tensor_helpers, …).
    fn generate_std_helpers(&self) -> String {
        r#""""
Std Helpers

Helper functions for ordinary std operators (array/nearest, object/get).
Ported from the TS shell (operator.rs) and the runtime evaluator.
"""

import math
import time
import datetime
from typing import Any, Dict, List, Optional


def array_nearest(query: List[float], candidates: List[List[float]]) -> Dict[str, Any]:
    """Cosine-nearest match: returns {index, score, margin}.

    Mirrors the TS `array/nearest` inline (operator.rs): cosine score per
    candidate, best index, and the margin between best and second-best
    (0 when only one candidate). `margin` uses -inf for the absent
    second-best so a single-candidate match reports margin 0.
    """
    if not candidates:
        raise ValueError("array/nearest: candidates array is empty, no nearest match exists")
    q_norm = math.sqrt(sum(x * x for x in query)) or 1.0
    scores: List[float] = []
    for c in candidates:
        c_norm = math.sqrt(sum(x * x for x in c)) or 1.0
        dot = sum(a * b for a, b in zip(query, c))
        scores.append(dot / (q_norm * c_norm))
    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    second = float("-inf")
    for i in range(len(scores)):
        if i != best_idx and scores[i] > second:
            second = scores[i]
    margin = 0.0 if len(scores) == 1 else best_score - second
    return {"index": best_idx, "score": best_score, "margin": margin}


def object_get(obj: Any, path: str, default: Any = None) -> Any:
    """Nested key-path access mirroring TS `object/get` (dot-split reduce)."""
    cur = obj
    for key in str(path).split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def object_merge(*dicts: Any) -> Dict[str, Any]:
    """Merge dicts left-to-right (later wins) — mirrors TS `Object.assign({}, …)`."""
    out: Dict[str, Any] = {}
    for d in dicts:
        if isinstance(d, dict):
            out.update(d)
    return out


def object_set(obj: Any, path: str, value: Any) -> Dict[str, Any]:
    """Return a copy of `obj` with `path` set to `value` (dot-path).

    Mirrors TS `object/set` — immutable update (does not mutate the input).
    """
    out = dict(obj) if isinstance(obj, dict) else {}
    keys = str(path).split(".")
    cur = out
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value
    return out


def array_filter(arr: Any, pred: Any) -> list:
    """Filter supporting callable and object-form predicates — TS `array/filter`."""
    src = arr if isinstance(arr, list) else list(arr or [])
    if callable(pred):
        return [x for x in src if pred(x)]
    if isinstance(pred, dict):
        return [
            x for x in src
            if isinstance(x, dict) and all(x.get(k) == v for k, v in pred.items())
        ]
    return src


def array_max(arr: Any, key: Any = None) -> float:
    """Max of array, or of a field across rows — TS `array/max` (with key)."""
    src = arr if isinstance(arr, list) else list(arr or [])
    if key is None:
        return max(src) if src else 0
    vals = [object_get(v, key) or 0 for v in src]
    return max(vals) if vals else 0


def array_min(arr: Any, key: Any = None) -> float:
    """Min of array, or of a field across rows — TS `array/min` (with key)."""
    src = arr if isinstance(arr, list) else list(arr or [])
    if key is None:
        return min(src) if src else 0
    vals = [object_get(v, key) or 0 for v in src]
    return min(vals) if vals else 0


def array_group(arr: Any, key: Any = None) -> Dict[Any, list]:
    """Group rows by a field/key-fn into {key: [rows]} — `array/group`."""
    src = arr if isinstance(arr, list) else list(arr or [])
    out: Dict[Any, list] = {}
    for v in src:
        if callable(key):
            k = key(v)
        else:
            k = object_get(v, key)
        out.setdefault(k, []).append(v)
    return out


def array_flatten(arr: Any, depth: int = 1) -> list:
    """Flatten a nested list up to `depth` levels — TS `array/flatten`."""
    def _flat(a: list, d: int) -> list:
        out: list = []
        for item in a:
            if isinstance(item, list) and d > 0:
                out.extend(_flat(item, d - 1))
            else:
                out.append(item)
        return out
    src = arr if isinstance(arr, list) else list(arr or [])
    try:
        d = int(depth)
    except (TypeError, ValueError):
        d = 1
    return _flat(src, d)


def grid_manhattan(p0: Any, p1: Any) -> float:
    """Manhattan distance between {x,y} cells — TS `grid/manhattan-distance`."""
    return abs((p0 or {}).get("x", 0) - (p1 or {}).get("x", 0)) + \
        abs((p0 or {}).get("y", 0) - (p1 or {}).get("y", 0))


def grid_chebyshev(p0: Any, p1: Any) -> float:
    """Chebyshev distance between {x,y} cells — TS `grid/distance`."""
    return max(abs((p0 or {}).get("x", 0) - (p1 or {}).get("x", 0)),
               abs((p0 or {}).get("y", 0) - (p1 or {}).get("y", 0)))


def grid_in_bounds(cell: Any, size: Any, height: Any = None) -> bool:
    """Cell {x,y} in a w×h grid — TS `grid/in-bounds`.

    `size` may be a scalar (square) or a {w,h} dict; `height` is the explicit
    height when three args are passed.
    """
    x = (cell or {}).get("x", 0)
    y = (cell or {}).get("y", 0)
    if height is not None:
        w, h = size, height
    elif isinstance(size, dict):
        w, h = size.get("w", 0), size.get("h", size.get("width", 0))
    else:
        w = h = size
    return x >= 0 and x < w and y >= 0 and y < h


# ---- time/* helpers (epoch-millis model, mirrors TS operator.rs) ----

_TIME_UNITS = {
    "seconds": 1000, "second": 1000,
    "minutes": 60000, "minute": 60000,
    "hours": 3600000, "hour": 3600000,
    "days": 86400000, "day": 86400000,
    "week": 604800000, "weeks": 604800000,
    "month": 2592000000, "months": 2592000000,
    "year": 31536000000, "years": 31536000000,
}


def time_unit_ms(unit: Any) -> int:
    """Millisecond multiplier for a time unit string — TS `time/add` map."""
    return _TIME_UNITS.get(str(unit), 1)


def time_parse(value: Any) -> int:
    """Parse an ISO/date string (or passthrough epoch-ms int) to epoch-ms."""
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value)
    try:
        return int(float(s))
    except ValueError:
        pass
    # ISO 8601 (trailing Z tolerated)
    iso = s.replace("Z", "+00:00")
    try:
        return int(datetime.datetime.fromisoformat(iso).timestamp() * 1000)
    except ValueError:
        return 0


def _from_ms(ms: Any) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(time_parse(ms) / 1000)


def time_format(ms: Any, fmt: Any = None) -> str:
    """Format epoch-ms; default locale date, else a strftime pattern."""
    if fmt:
        return _from_ms(ms).strftime(str(fmt))
    return _from_ms(ms).strftime("%Y-%m-%d")


def time_year(ms: Any) -> int:
    return _from_ms(ms).year


def time_month(ms: Any) -> int:
    return _from_ms(ms).month


def time_day(ms: Any) -> int:
    return _from_ms(ms).day


def time_hour(ms: Any) -> int:
    return _from_ms(ms).hour


def time_minute(ms: Any) -> int:
    return _from_ms(ms).minute


def time_second(ms: Any) -> int:
    return _from_ms(ms).second


def time_weekday(ms: Any) -> int:
    return _from_ms(ms).weekday()
"#.to_string()
    }

    /// Probabilistic-operator helpers (Phase 4). Semantics ported from the
    /// runtime evaluator `packages/almadar-evaluator/std/prob.ts`.
    fn generate_prob_helpers(&self) -> String {
        r#""""
Prob Helpers

Statistical helpers for prob/* operators. Ported from the runtime
path (packages/almadar-evaluator/std/prob.ts) so the compiled Python
shell matches the JS interpreter's semantics.
"""

import math
import random
from typing import Dict, List


def prob_poisson(lmbda: float) -> int:
    """Knuth's algorithm — mirrors evalProbPoisson."""
    limit = math.exp(-lmbda)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if not (p > limit):
            break
    return k - 1


def prob_expected_value(samples: List[float]) -> float:
    """Mean of numeric samples; 0 on empty — mirrors evalProbExpectedValue."""
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def prob_variance(samples: List[float]) -> float:
    """Population variance; 0 on empty — mirrors evalProbVariance."""
    if not samples:
        return 0.0
    mean = sum(samples) / len(samples)
    return sum((v - mean) ** 2 for v in samples) / len(samples)


def prob_percentile(samples: List[float], p: float) -> float:
    """p-th percentile (0-100) with linear interpolation — evalProbPercentile."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = (p / 100.0) * (len(ordered) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def prob_credible_interval(samples: List[float], alpha: float) -> List[float]:
    """Symmetric credible interval [lo, hi] at alpha — evalProbCredibleInterval.

    Uses length-1 indexing (matching the runtime: floor(alpha/2*(n-1)) /
    ceil((1-alpha/2)*(n-1))), NOT length indexing.
    """
    if not samples:
        return [0.0, 0.0]
    ordered = sorted(samples)
    lo_idx = math.floor((alpha / 2.0) * (len(ordered) - 1))
    hi_idx = math.ceil((1.0 - alpha / 2.0) * (len(ordered) - 1))
    return [ordered[lo_idx], ordered[hi_idx]]


def prob_histogram(samples: List[float], bins: int) -> Dict[str, list]:
    """Bin numeric samples — mirrors evalProbHistogram."""
    if not samples:
        return {"binEdges": [], "counts": []}
    lo = min(samples)
    hi = max(samples)
    bin_width = (hi - lo) / bins or 1.0
    bin_edges = [lo + i * bin_width for i in range(bins + 1)]
    counts = [0] * bins
    for s in samples:
        idx = int((s - lo) / bin_width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return {"binEdges": bin_edges, "counts": counts}
"#.to_string()
    }
}

/// Implement the compiler's Backend trait
impl Backend for PythonBackend {
    fn generate(&self, module: &OirModule) -> GeneratedOutput {
        let files = self.generate_internal(module);
        GeneratedOutput {
            files,
            warnings: vec![],
        }
    }

    fn generate_with_mode(
        &self,
        module: &OirModule,
        mode: orbital_core::generate::ShellMode,
    ) -> GeneratedOutput {
        use orbital_core::generate::ShellMode;

        let mut files = Vec::new();

        match mode {
            ShellMode::Full => {
                // Generate everything
                return self.generate(module);
            }
            ShellMode::Server => {
                // Server-only: entities, traits, routes, helpers (no tests, no pages)

                // 1. Generate entity models (needed for API)
                files.push(GeneratedFile::new(
                    "entities/__init__.py",
                    generate_entities(module),
                ));

                // 2. Generate trait state machines
                let trait_files = generate_traits(module);
                for (filename, content) in trait_files {
                    files.push(GeneratedFile::new(format!("traits/{}", filename), content));
                }
                files.push(GeneratedFile::new(
                    "traits/__init__.py",
                    self.generate_traits_init(module),
                ));

                // 3. Generate API routes
                for trait_def in &module.traits {
                    files.push(GeneratedFile::new(
                        format!("routes/{}.py", to_snake_case(&trait_def.name)),
                        self.generate_trait_routes(trait_def, module),
                    ));
                }
                files.push(GeneratedFile::new(
                    "routes/__init__.py",
                    self.generate_routes_init(module),
                ));

                // 4. Generate helpers (needed for NN/tensor ops)
                files.push(GeneratedFile::new(
                    "nn_helpers.py",
                    self.generate_nn_helpers(),
                ));
                files.push(GeneratedFile::new(
                    "tensor_helpers.py",
                    self.generate_tensor_helpers(),
                ));
                files.push(GeneratedFile::new(
                    "train_helpers.py",
                    self.generate_train_helpers(),
                ));
                files.push(GeneratedFile::new(
                    "effect_helpers.py",
                    self.generate_effect_helpers(),
                ));
                files.push(GeneratedFile::new(
                    "graph_helpers.py",
                    self.generate_graph_helpers(),
                ));
                files.push(GeneratedFile::new(
                    "data_helpers.py",
                    self.generate_data_helpers(),
                ));
                files.push(GeneratedFile::new(
                    "contract_helpers.py",
                    self.generate_contract_helpers(),
                ));
                files.push(GeneratedFile::new(
                    "ml_effect_helpers.py",
                    self.generate_ml_effect_helpers(),
                ));

                // 5. Generate __init__.py
                files.push(GeneratedFile::new(
                    "__init__.py",
                    self.generate_main_init(module),
                ));

                // 6. Generate tests (exhaustion tests are critical for backend)
                let test_files = generate_tests(module);
                for test_file in test_files {
                    files.push(GeneratedFile::new(test_file.path, test_file.content));
                }
            }
            ShellMode::Client => {
                // Client-only: entities only (for type definitions)
                // Python client mode is minimal - just types

                files.push(GeneratedFile::new(
                    "entities/__init__.py",
                    generate_entities(module),
                ));

                files.push(GeneratedFile::new(
                    "__init__.py",
                    "\"\"\"Generated Orbital types.\"\"\"\n\nfrom .entities import *\n".to_string(),
                ));
            }
        }

        GeneratedOutput {
            files,
            warnings: vec![],
        }
    }

    fn name(&self) -> &'static str {
        "python"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        let backend = PythonBackend::new();
        assert_eq!(backend.name(), "python");
    }

    #[test]
    fn test_to_snake_case() {
        use orbital_core::str_utils::to_snake_case;
        assert_eq!(to_snake_case("HelloWorld"), "hello_world");
        assert_eq!(to_snake_case("NeuralProposer"), "neural_proposer");
    }

    /// Verify that the emitted route file is valid Python syntax.
    ///
    /// Generates routes for a minimal runtime trait and runs
    /// `python3 -m py_compile` so the compiler gate catches syntax regressions
    /// without requiring a full FastAPI install.
    #[test]
    fn test_generated_routes_are_valid_python() {
        use orbital_core::ir::{OirEntity, OirEvent, OirModule, OirPersistence, OirState, OirTrait};

        let mut entity = OirEntity::new("Config");
        entity.persistence = OirPersistence::Runtime;

        let mut idle = OirState::new("idle");
        idle.is_initial = true;
        let active = OirState::new("active");

        let mut trait_def = OirTrait::new("ConfigTrait");
        trait_def.linked_entity = Some("Config".to_string());
        trait_def.states = vec![idle, active];
        trait_def.events = vec![OirEvent::new("ACTIVATE")];

        let mut module = OirModule::new("TestApp", "1.0.0");
        module.entities.push(entity);
        module.traits.push(trait_def.clone());

        let backend = PythonBackend::new();
        let route_code = backend.generate_trait_routes(&trait_def, &module);

        // Write to a temp file and syntax-check with python3
        let tmp = std::env::temp_dir().join("orbital_py_route_test.py");
        std::fs::write(&tmp, &route_code).expect("write temp file");
        let status = std::process::Command::new("python3")
            .args(["-m", "py_compile", tmp.to_str().unwrap()])
            .status()
            .expect("python3 not found");
        assert!(status.success(), "generated route file has syntax errors:\n{}", route_code);
    }
}
