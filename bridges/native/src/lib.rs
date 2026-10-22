//! The kernel as a native Python extension. `Server` is the `orbital-wasm` server API (`orbital_wasm::stored`,
//! `api`), the same surface `orbital-node` exports, so a host loads any of them behind the same ports. Marshalling
//! only: every decision is `orbital_core::embed`'s. The GIL is released while the kernel runs.

use orbital_wasm::api;
use orbital_wasm::stored::{ServerHandle, Slot};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

fn fail(message: String) -> PyErr {
    PyRuntimeError::new_err(message)
}

/// The retired embedded HTTP adapter. Native kernel APIs below remain available.
#[pyclass]
struct NativeServer {
    unavailable: std::convert::Infallible,
}

#[pymethods]
impl NativeServer {
    #[staticmethod]
    #[pyo3(signature = (program_json, env_json, catalogs_json, on_live=None))]
    fn open(program_json: &str, env_json: &str, catalogs_json: &str, on_live: Option<PyObject>) -> PyResult<Self> {
        let _ = (program_json, env_json, catalogs_json, on_live);
        Err(fail("NATIVE_SERVER_UNAVAILABLE: the embedded HTTP adapter retired with orbital-server; the std server adapter is pending. Native kernel Server APIs remain available.".to_string()))
    }

    fn handle(
        &self,
        py: Python<'_>,
        method: &str,
        uri: &str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> PyResult<(u16, Vec<(String, String)>, Vec<u8>)> {
        let _ = (py, method, uri, headers, body);
        match self.unavailable {}
    }

    #[pyo3(signature = (authorization=None))]
    fn viewer(&self, py: Python<'_>, authorization: Option<&str>) -> PyResult<String> {
        let _ = (py, authorization);
        match self.unavailable {}
    }

    fn state(&self, trait_name: &str) -> String {
        let _ = trait_name;
        match self.unavailable {}
    }

    fn take_ticket(&self, ticket: &str) -> Option<String> {
        let _ = ticket;
        match self.unavailable {}
    }

    fn live_recipients(&self, item_json: &str, connections_json: &str) -> PyResult<String> {
        let _ = (item_json, connections_json);
        match self.unavailable {}
    }
}

/// The server role: `start` a request, answer the I/O it asks for, `resume` the flight until it is done.
#[pyclass]
struct Server {
    held: Slot<ServerHandle>,
}

impl Server {
    fn handle(&self) -> PyResult<Arc<ServerHandle>> {
        self.held.get().map_err(fail)
    }
}

#[pymethods]
impl Server {
    #[new]
    fn new(resolved_schema_json: &str) -> PyResult<Self> {
        Ok(Self {
            held: Slot::new(ServerHandle::plain(resolved_schema_json).map_err(fail)?),
        })
    }

    /// A server that answers its own row asks from the store `env_json` selects (`DATA_BACKEND`, `DATABASE_URL`,
    /// `COUCHDB_URL`, the Firebase variables); the host answers the rest.
    #[staticmethod]
    fn with_store(resolved_schema_json: &str, env_json: &str) -> PyResult<Self> {
        let env: HashMap<String, String> =
            serde_json::from_str(env_json).map_err(|e| fail(format!("parse env_json: {e}")))?;
        Ok(Self {
            held: Slot::new(ServerHandle::from_env(resolved_schema_json, &env).map_err(fail)?),
        })
    }

    fn free(&self) {
        self.held.take();
    }

    fn os_allows(&self, requirement_json: &str) -> PyResult<bool> {
        self.handle()?
            .api()
            .os_allows(requirement_json)
            .map_err(fail)
    }

    fn start(&self, py: Python<'_>, orbital: &str, request_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.start(orbital, request_json))
            .map_err(fail)
    }

    fn resume(&self, py: Python<'_>, flight: u64, replies_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.resume(flight, replies_json))
            .map_err(fail)
    }

    fn resolve_rows(&self, py: Python<'_>, ref_json: &str, viewer_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.resolve_rows(ref_json, viewer_json))
            .map_err(fail)
    }

    fn resume_resolve(&self, py: Python<'_>, flight: u64, replies_json: &str) -> PyResult<String> {
        let handle = self.handle()?;
        py.allow_threads(|| handle.resume_resolve(flight, replies_json))
            .map_err(fail)
    }

    fn abandon(&self, flight: u64) -> PyResult<bool> {
        Ok(self.handle()?.abandon(flight))
    }

    fn in_flight(&self) -> PyResult<usize> {
        Ok(self.handle()?.api().in_flight())
    }

    #[pyo3(signature = (trait_name, entity_id=None))]
    fn get_state(&self, trait_name: &str, entity_id: Option<&str>) -> PyResult<String> {
        Ok(self.handle()?.api().get_state(trait_name, entity_id))
    }

    fn set_managed_program(&self, program_json: &str) -> PyResult<()> {
        self.handle()?.api().set_managed_program(program_json).map_err(fail)
    }

    fn set_default_user(&self, user_json: &str) -> PyResult<()> {
        self.handle()?
            .api()
            .set_default_user(user_json)
            .map_err(fail)
    }

    fn mount(&self, traits_json: &str) -> PyResult<()> {
        self.handle()?.api().mount(traits_json).map_err(fail)
    }

    fn remount(&self, traits_json: &str) -> PyResult<()> {
        self.handle()?.api().remount(traits_json).map_err(fail)
    }

    fn set_messages(&self, locales_json: &str, catalogs_json: &str) -> PyResult<()> {
        self.handle()?
            .api()
            .set_messages(locales_json, catalogs_json)
            .map_err(fail)
    }

    fn set_capabilities(
        &self,
        entity_type: &str,
        query_json: &str,
        write_json: &str,
    ) -> PyResult<()> {
        self.handle()?
            .api()
            .set_capabilities(entity_type, query_json, write_json)
            .map_err(fail)
    }

    fn set_tick_mount(&self, traits_json: &str) -> PyResult<()> {
        self.handle()?
            .api()
            .set_tick_mount(traits_json)
            .map_err(fail)
    }

    fn last_error(&self) -> PyResult<String> {
        Ok(self.handle()?.api().last_error())
    }
}

#[pyfunction]
fn install_language(bundle_json: &str) -> PyResult<()> {
    api::install_language(bundle_json).map_err(fail)
}

#[pyfunction]
fn os_glob_matches(glob: &str, path: &str) -> bool {
    api::os_glob_matches(glob, path)
}

#[pymodule]
fn orbital_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    m.add_class::<NativeServer>()?;
    m.add_function(wrap_pyfunction!(install_language, m)?)?;
    m.add_function(wrap_pyfunction!(os_glob_matches, m)?)?;
    Ok(())
}
