//! The kernel as a native Python extension. `Server` is the `orbital-wasm` server API (`orbital_wasm::stored`,
//! `api`), the same surface `orbital-node` exports, so a host loads any of them behind the same ports. Marshalling
//! only: every decision is `orbital_core::embed`'s. The GIL is released while the kernel runs.

use orbital_wasm::api;
use orbital_wasm::stored::{ServerHandle, Slot};
use pyo3::exceptions::{PyPermissionError, PyRuntimeError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

fn fail(message: String) -> PyErr {
    PyRuntimeError::new_err(message)
}

/// A program served by `orbital-server` in this process: every request goes through the server's own router, so a
/// Python host serves the routes `orb serve` serves. The host keeps its listener and the live connections.
#[pyclass]
struct NativeServer {
    host: orbital_server::embedded::EmbeddedHost,
}

#[pymethods]
impl NativeServer {
    /// Boot `program_json` with the environment `env_json` and the message catalogs `catalogs_json`; `on_live(item_json)`
    /// receives each live item.
    #[staticmethod]
    #[pyo3(signature = (program_json, env_json, catalogs_json, on_live=None))]
    fn open(program_json: &str, env_json: &str, catalogs_json: &str, on_live: Option<PyObject>) -> PyResult<Self> {
        let env: HashMap<String, String> = serde_json::from_str(env_json).map_err(|e| fail(format!("parse env_json: {e}")))?;
        let catalogs: orbital_server::Catalogs =
            serde_json::from_str(catalogs_json).map_err(|e| fail(format!("parse catalogs_json: {e}")))?;
        let on_live: Box<dyn Fn(String) + Send + Sync> = Box::new(move |item_json| {
            let Some(callback) = on_live.as_ref() else { return };
            Python::with_gil(|py| {
                if let Err(error) = callback.call1(py, (item_json,)) {
                    error.print(py);
                }
            });
        });
        let host = orbital_server::embedded::EmbeddedHost::open(program_json, &env, catalogs, on_live).map_err(|e| fail(format!("{e:#}")))?;
        Ok(Self { host })
    }

    /// Run one request through the server's router: `(status, headers, body)`.
    fn handle(
        &self,
        py: Python<'_>,
        method: &str,
        uri: &str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> PyResult<(u16, Vec<(String, String)>, Vec<u8>)> {
        let response = py.allow_threads(|| self.host.handle(method, uri, &headers, body)).map_err(|e| fail(format!("{e:#}")))?;
        Ok((response.status, response.headers, response.body))
    }

    /// The viewer `authorization` names, as JSON (`null` when the program declares no sign-in); raises `PermissionError`
    /// for a bearer that does not verify.
    #[pyo3(signature = (authorization=None))]
    fn viewer(&self, py: Python<'_>, authorization: Option<&str>) -> PyResult<String> {
        py.allow_threads(|| self.host.viewer(authorization)).map_err(|e| PyPermissionError::new_err(e))
    }

    fn state(&self, trait_name: &str) -> String {
        self.host.state(trait_name)
    }

    fn take_ticket(&self, ticket: &str) -> Option<String> {
        self.host.take_ticket(ticket)
    }

    fn live_recipients(&self, item_json: &str, connections_json: &str) -> PyResult<String> {
        self.host.live_recipients(item_json, connections_json).map_err(fail)
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
